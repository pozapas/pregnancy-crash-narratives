#!/usr/bin/env Rscript
# Paper 2 Step 4 -- persons, denominators, and case covariates (§3, §4.3, §4.4).
#
# Reads the 8 GB / 11.25 M-row person-level fst IN CHUNKS (a whole-file read needs several GB
# and Paper 1's Jev screen runs concurrently on this machine). Produces three things:
#
#   1. persons_denominators.csv  -- Year x role x age-band counts of UNIQUE female persons,
#      the denominators for every rate in T4. Also male and all-person counts so the
#      female share can be sanity-checked against published CRIS totals.
#   2. female1549.fst -- one row per unique female person aged 15-49 with the §3 covariates,
#      the analysis frame for the §4.4 documentation model.
#   3. crash_female1549.fst -- crash-level roll-up (how many female 15-49 persons of each
#      role, worst injury among them, any transport proxy), for the crash-level sensitivity
#      version of the documentation model and for joining Stage-B cases.
#
# WHAT THIS FILE ACTUALLY IS (probed at five offsets spanning 2017-2025 before any counting):
# it is ONE ROW PER UNIT, not one row per person. `Prsn_Nbr` is 1 in every non-null row, and
# 97% of person rows are `Driver`; `Passenger/Occupant` is 0.4-0.6%, which is the handful of
# units whose reported person was not the driver. PASSENGERS ARE NOT IN THIS EXTRACT, and
# `Num_pr` tracks `Num_un` rather than true occupancy, so it cannot stand in for them.
#
# Consequence for §4.3, and it is a deviation from the outline: there is no passenger
# denominator to be had here. Denominators are therefore built ROLE-MATCHED -- female
# drivers 15-49 and female non-occupants (pedestrian/pedalcyclist) 15-49 -- and the rate in
# T4 is reported per 1,000 female DRIVERS 15-49 against Jev cases with `preg_role = driver`.
# Pregnant-passenger cases are reported as counts with no rate, and §7 says so. A matched
# numerator and denominator is the stricter comparison anyway; what is lost is passenger rates.
#
# DE-DUPLICATION: every count below is over unique `Person_ID`
# (Crash_ID_Pr _ UnitNbr_Pr _ Prsn_Nbr), never over rows; the probe measured
# rows/uniqueN(Person_ID) = 1.0000, and the full pass re-checks it. Rows with a NA
# Person_ID (8.8% -- unit rows carrying no person record) are dropped from person counts.
#
# Usage:
#   Rscript p04_persons.R --probe          # value distributions + dedup check, one chunk
#   Rscript p04_persons.R [--chunk 400000]

suppressMessages({library(fst); library(data.table)})

FST  <- file.path(Sys.getenv("CRIS_DATA_DIR"), "CrUnPPr_2017_2025_LimWithNarr2_95VarV01.fst")
JEVROOT <- Sys.getenv("JEV_ROOT", unset = normalizePath(file.path("..", "..")))
OUT <- file.path(JEVROOT, "paper2/data/persons")
args  <- commandArgs(trailingOnly = TRUE)
PROBE <- "--probe" %in% args
CHUNK <- if ("--chunk" %in% args) as.integer(args[which(args == "--chunk") + 1]) else 400000L
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

COLS <- c("Crash_ID", "Year", "Cnty_ID", "Crash_Sev_ID", "Crash_Speed_Limit", "Crash_Time",
          "Day_of_Week", "Rural_Urban_Type_ID", "FHE_Collsn_ID", "Harm_Evnt_ID",
          "Tot_Injry_Cnt", "Death_Cnt", "Rural_Fl", "Pop_Group_ID", "Onsys_Fl",
          "Num_un", "Num_pr",
          "Unit_Desc_ID", "Veh_Body_Styl_ID", "Veh_Mod_Year", "Emer_Respndr_Fl",
          "Crash_ID_Pr", "UnitNbr_Pr", "Prsn_Nbr", "Person_ID", "Prsn_Type_ID",
          "Prsn_Occpnt_Pos_ID", "Prsn_Injry_Sev_ID", "Prsn_Age", "Prsn_Gndr_ID",
          "Prsn_Ejct_ID", "Prsn_Rest_ID", "Prsn_Airbag_ID")

meta  <- fst::metadata_fst(FST)
NROW  <- meta$nrOfRows
cat(sprintf("fst rows: %s  chunk: %s\n", format(NROW, big.mark = ","), format(CHUNK, big.mark = ",")))

# ---------------------------------------------------------------- derivations
ROLE_MAP <- function(x) {
  x <- tolower(trimws(as.character(x)))
  fifelse(x %chin% c("driver", "driver of motorcycle type vehicle"), "driver",
  fifelse(grepl("^passenger", x), "passenger",
  fifelse(x == "pedestrian", "pedestrian",
  fifelse(x == "pedalcyclist", "pedalcyclist", "other_unknown"))))
}

prep <- function(d) {
  for (cc in c("Prsn_Gndr_ID", "Prsn_Type_ID", "Prsn_Rest_ID", "Prsn_Airbag_ID",
               "Prsn_Ejct_ID", "Prsn_Injry_Sev_ID", "Crash_Sev_ID", "Rural_Urban_Type_ID",
               "Emer_Respndr_Fl", "Unit_Desc_ID", "Veh_Body_Styl_ID", "FHE_Collsn_ID",
               "Cnty_ID", "Day_of_Week", "Year"))
    set(d, j = cc, value = trimws(as.character(d[[cc]])))

  # Age sentinels: CRIS uses 99 / 999 for unknown in some feeds; anything outside 0-109 is
  # treated as missing rather than silently entering an age filter.
  d[, age := as.numeric(Prsn_Age)]
  d[!is.na(age) & (age < 0 | age > 109 | age %in% c(999)), age := NA_real_]

  d[, role := ROLE_MAP(Prsn_Type_ID)]
  d[, female := Prsn_Gndr_ID == "Female"]
  d[, f1549 := female & !is.na(age) & age >= 15 & age <= 49]
  d[, age_band := fifelse(is.na(age), "unknown",
                   fifelse(age < 15, "0-14",
                   fifelse(age <= 19, "15-19",
                   fifelse(age <= 24, "20-24",
                   fifelse(age <= 29, "25-29",
                   fifelse(age <= 34, "30-34",
                   fifelse(age <= 39, "35-39",
                   fifelse(age <= 44, "40-44",
                   fifelse(age <= 49, "45-49", "50+")))))))))]
  d[, unbelted := Prsn_Rest_ID == "None"]
  d[, airbag_deployed := grepl("^Deployed", Prsn_Airbag_ID)]
  d[, ejected := grepl("^Yes", Prsn_Ejct_ID)]   # "Yes" and "Yes, Partial" are the only ejected codes
  d[, injured := !(Prsn_Injry_Sev_ID %chin% c("Not Injured", "Unknown", "", NA))]
  d[, killed := Prsn_Injry_Sev_ID == "Killed"]
  d[, susp_serious := Prsn_Injry_Sev_ID %chin% c("Incapacitating Injury", "Suspected Serious Injury")]
  d[, emer_resp := Emer_Respndr_Fl == "Y"]
  # Rural_Urban_Type_ID is NA on ~50% of rows; Rural_Fl is complete, so it is the primary
  # rural measure and Rural_Urban_Type_ID is kept as the finer stratum where present.
  d[, rural := fifelse(!is.na(Rural_Fl) & trimws(as.character(Rural_Fl)) %chin% c("Y", "1", "TRUE"), TRUE,
               fifelse(!is.na(Rural_Urban_Type_ID) & grepl("Rural", Rural_Urban_Type_ID), TRUE, FALSE))]
  d[, hour := suppressWarnings(as.integer(substr(Crash_Time, 1, 2)))]
  d[, ampm := toupper(substr(Crash_Time, nchar(Crash_Time) - 1, nchar(Crash_Time)))]
  d[, hour24 := fifelse(is.na(hour), NA_integer_,
                 fifelse(ampm == "AM", fifelse(hour == 12L, 0L, hour),
                         fifelse(hour == 12L, 12L, hour + 12L)))]
  d[]
}

# ---------------------------------------------------------------- probe
if (PROBE) {
  d <- as.data.table(read_fst(FST, columns = COLS, from = 1, to = CHUNK))
  cat(sprintf("\nrows=%s  uniqueN(Person_ID)=%s  NA Person_ID=%s\n",
              format(nrow(d), big.mark = ","),
              format(uniqueN(d$Person_ID), big.mark = ","),
              format(sum(is.na(d$Person_ID)), big.mark = ",")))
  dn <- d[!is.na(Person_ID)]
  cat(sprintf("non-NA rows=%s  unique=%s  ratio=%.4f\n",
              format(nrow(dn), big.mark = ","), format(uniqueN(dn$Person_ID), big.mark = ","),
              nrow(dn) / uniqueN(dn$Person_ID)))
  cat("\nNum_pr vs observed persons per crash (first 5 crashes):\n")
  print(head(d[!is.na(Person_ID), .(observed = uniqueN(Person_ID)), by = Crash_ID], 5))
  d <- prep(d)
  for (cc in c("Prsn_Gndr_ID", "Prsn_Type_ID", "role", "age_band", "Prsn_Rest_ID",
               "Prsn_Airbag_ID", "Prsn_Ejct_ID", "Prsn_Injry_Sev_ID", "Rural_Urban_Type_ID",
               "Emer_Respndr_Fl")) {
    cat(sprintf("\n-- %s --\n", cc)); print(sort(table(d[[cc]], useNA = "ifany"), decreasing = TRUE)[1:12])
  }
  cat("\n-- age summary --\n"); print(summary(d$age))
  cat(sprintf("\nfemale 15-49 unique persons in chunk: %s\n",
              format(uniqueN(d[f1549 == TRUE]$Person_ID), big.mark = ",")))
  quit(status = 0)
}

# ---------------------------------------------------------------- full pass
denom  <- list()
fem    <- list()
crashf <- list()
seen   <- new.env(hash = TRUE, parent = emptyenv())   # not used; dedup is done at the end
t0 <- Sys.time(); i <- 0L; from <- 1L

while (from <= NROW) {
  to <- min(from + CHUNK - 1L, NROW)
  i <- i + 1L
  d <- as.data.table(read_fst(FST, columns = COLS, from = from, to = to))
  d <- d[!is.na(Person_ID)]
  if (nrow(d)) {
    d <- prep(d)
    # Chunk-local dedup; a person's rows are contiguous (same crash), so cross-chunk
    # duplicates can only occur at a boundary and are removed in the final fold.
    d <- unique(d, by = "Person_ID")

    denom[[i]] <- d[, .(n = .N,
                        n_injured = sum(injured, na.rm = TRUE),
                        n_killed  = sum(killed,  na.rm = TRUE)),
                    by = .(Year, role, Prsn_Gndr_ID, age_band)]

    f <- d[f1549 == TRUE, .(Person_ID, Crash_ID, Crash_ID_Pr, UnitNbr_Pr, Prsn_Nbr, Year,
                            Cnty_ID, Crash_Sev_ID, Crash_Speed_Limit, Day_of_Week, hour24,
                            Rural_Urban_Type_ID, rural, FHE_Collsn_ID, Harm_Evnt_ID,
                            Tot_Injry_Cnt, Death_Cnt, Num_un, Num_pr, Pop_Group_ID,
                            Onsys_Fl, Unit_Desc_ID, Veh_Body_Styl_ID,
                            Veh_Mod_Year, emer_resp, role, Prsn_Occpnt_Pos_ID,
                            Prsn_Injry_Sev_ID, age, age_band, Prsn_Rest_ID, unbelted,
                            Prsn_Airbag_ID, airbag_deployed, ejected, injured, killed,
                            susp_serious)]
    if (nrow(f)) fem[[i]] <- f

    crashf[[i]] <- d[f1549 == TRUE, .(
      n_f1549            = .N,
      n_f1549_driver     = sum(role == "driver"),
      n_f1549_passenger  = sum(role == "passenger"),
      n_f1549_ped        = sum(role %chin% c("pedestrian", "pedalcyclist")),
      f1549_any_injured  = any(injured, na.rm = TRUE),
      f1549_any_serious  = any(susp_serious | killed, na.rm = TRUE),
      f1549_any_unbelted = any(unbelted, na.rm = TRUE),
      f1549_min_age      = min(age, na.rm = TRUE),
      f1549_max_age      = max(age, na.rm = TRUE)), by = .(Crash_ID, Year)]
  }
  if (i %% 5L == 0L)
    cat(sprintf("  chunk %d  rows %s..%s  %.0fs\n", i, format(from, big.mark = ","),
                format(to, big.mark = ","), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  from <- to + 1L
  rm(d); gc(verbose = FALSE)
}

# ---- fold, with a final global dedup across chunk boundaries -------------------
FEM <- unique(rbindlist(fem, use.names = TRUE), by = "Person_ID")
fwrite(FEM, file.path(OUT, "female1549.csv"))
write_fst(FEM, file.path(OUT, "female1549.fst"), compress = 70)

DEN <- rbindlist(denom)[, .(n = sum(n), n_injured = sum(n_injured), n_killed = sum(n_killed)),
                        by = .(Year, role, Prsn_Gndr_ID, age_band)]
fwrite(DEN, file.path(OUT, "persons_denominators.csv"))

CR <- rbindlist(crashf)[, .(
  n_f1549 = sum(n_f1549), n_f1549_driver = sum(n_f1549_driver),
  n_f1549_passenger = sum(n_f1549_passenger), n_f1549_ped = sum(n_f1549_ped),
  f1549_any_injured = any(f1549_any_injured), f1549_any_serious = any(f1549_any_serious),
  f1549_any_unbelted = any(f1549_any_unbelted),
  f1549_min_age = min(f1549_min_age), f1549_max_age = max(f1549_max_age)),
  by = .(Crash_ID, Year)]
write_fst(CR, file.path(OUT, "crash_female1549.fst"), compress = 70)
fwrite(CR, file.path(OUT, "crash_female1549.csv"))

# ---- headline denominator table: unique female persons 15-49 per Year x role ---
F1549 <- DEN[Prsn_Gndr_ID == "Female" & age_band %chin%
               c("15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"),
             .(n_female_15_49 = sum(n)), by = .(Year, role)]
fwrite(F1549, file.path(OUT, "denominator_female1549_by_year_role.csv"))

summ <- list(
  fst_rows = NROW,
  chunks = i,
  female1549_persons = nrow(FEM),
  crashes_with_female1549 = nrow(CR),
  denominator_rows = nrow(DEN),
  female1549_by_year = as.list(FEM[, .N, by = Year][order(Year)]),
  female1549_by_role = as.list(FEM[, .N, by = role][order(-N)]),
  seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
writeLines(jsonlite::toJSON(summ, auto_unbox = TRUE, pretty = TRUE),
           file.path(OUT, "persons_summary.json"))
cat("\nSTEP 4 DONE\n")
print(summ[c("female1549_persons", "crashes_with_female1549", "seconds")])
