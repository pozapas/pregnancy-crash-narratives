#!/usr/bin/env Rscript
# Paper 2 -- join Stage-B confirmed cases to coded CRIS fields, and build the §4.4
# documentation-model frame.
#
# Two outputs, two units of analysis:
#
#   cases_covariates.fst   one row per CONFIRMED pregnancy crash (Stage-B p > tau), carrying
#                          the crash-level covariates plus, where the Jev role is `driver`,
#                          the attributes of the female 15-49 driver(s) in that crash. This is
#                          the frame for T5, T7, F4, F5, F7.
#
#   docmodel_frame.fst     the §4.4 analysis frame: a female 15-49 DRIVER person is eligible
#                          when she is the ONLY female 15-49 driver in her crash, and y = 1 if
#                          that crash is a confirmed case with Jev role `driver`.
#
#                          The restriction is not cosmetic. A narrative documents that "the
#                          driver was pregnant"; it does not say WHICH driver when a two-vehicle
#                          crash has two female drivers aged 15-49. Flagging both attaches the
#                          pregnancy to a person whose belt use, airbag and injury are someone
#                          else's roughly a quarter of the time, which is exactly the person-level
#                          covariate the model is about. So ambiguous crashes leave the frame on
#                          BOTH sides -- cases and controls -- keeping the eligible population a
#                          single well-defined set: female 15-49 drivers who are the only such
#                          person in their crash. `y_any` retains the permissive coding for a
#                          §5 sensitivity fit, and the counts of what was dropped are recorded.
#
#                          Controls are then sampled (seed 7, default 250k).
#                          Under case-control sampling of controls at a known rate, the fitted
#                          odds ratios are unbiased and only the intercept is shifted; the
#                          sampling fraction is stored so the intercept can be corrected and
#                          so predicted probabilities can be rescaled. `w` holds the inverse
#                          sampling weight for anyone who prefers a weighted fit.
#
# WHY DRIVERS. p04_persons.R established that this extract is one row per unit, so passengers
# are absent and no passenger denominator exists. The documentation model is therefore fitted
# on female 15-49 drivers, where numerator and denominator are matched. 31% of confirmed cases
# are passengers; they are described in T7 as counts and carry no rate. §7 states this.
#
# Usage: Rscript p06_cases_join.R [--tau 0.5] [--controls 250000]

suppressMessages({library(fst); library(data.table)})

JEVROOT <- Sys.getenv("JEV_ROOT", unset = normalizePath(file.path("..", "..")))
P2  <- file.path(JEVROOT, "paper2")
args <- commandArgs(trailingOnly = TRUE)
TAU  <- if ("--tau" %in% args) as.numeric(args[which(args == "--tau") + 1]) else 0.5
NCTRL <- if ("--controls" %in% args) as.integer(args[which(args == "--controls") + 1]) else 250000L
SEED <- 7L
set.seed(SEED)

# ---- Stage-B confirmed cases ---------------------------------------------------
sb <- fread(file.path(P2, "data/stageB/stageB_flat.csv"))
cat(sprintf("stageB rows: %s\n", format(nrow(sb), big.mark = ",")))
sb[, confirmed := preg_mentioned_p > TAU]
cases <- sb[confirmed == TRUE]
cat(sprintf("confirmed cases (p > %.2f): %s\n", TAU, format(nrow(cases), big.mark = ",")))

fem <- as.data.table(read_fst(file.path(P2, "data/persons/female1549.fst")))
crf <- as.data.table(read_fst(file.path(P2, "data/persons/crash_female1549.fst")))
cat(sprintf("female 15-49 persons: %s   crashes with one: %s\n",
            format(nrow(fem), big.mark = ","), format(nrow(crf), big.mark = ",")))

# ---- 1. case-level covariates --------------------------------------------------
# Crash-level attributes are taken from the first female-15-49 row of the crash where one
# exists (crash fields are constant within a crash), and from a driver row otherwise.
crash_cov <- unique(fem[, .(Crash_ID, Cnty_ID, Crash_Sev_ID, Crash_Speed_Limit, Day_of_Week,
                            hour24, Rural_Urban_Type_ID, rural, FHE_Collsn_ID, Harm_Evnt_ID,
                            Tot_Injry_Cnt, Death_Cnt, Num_un, Num_pr, Pop_Group_ID, Onsys_Fl,
                            emer_resp)], by = "Crash_ID")

# Person attributes of the female 15-49 person whose role matches the Jev-assigned role.
fem_drv <- fem[role == "driver", .(
  f_driver_n         = .N,
  f_driver_age       = min(age, na.rm = TRUE),
  f_driver_unbelted  = any(unbelted, na.rm = TRUE),
  f_driver_airbag    = any(airbag_deployed, na.rm = TRUE),
  f_driver_ejected   = any(ejected, na.rm = TRUE),
  f_driver_injured   = any(injured, na.rm = TRUE),
  f_driver_serious   = any(susp_serious | killed, na.rm = TRUE),
  f_driver_killed    = any(killed, na.rm = TRUE),
  f_driver_rest      = Prsn_Rest_ID[1],
  f_driver_airbag_id = Prsn_Airbag_ID[1],
  f_driver_injsev    = Prsn_Injry_Sev_ID[1],
  f_driver_body      = Veh_Body_Styl_ID[1],
  f_driver_modyear   = Veh_Mod_Year[1]), by = Crash_ID]
fem_ped <- fem[role %chin% c("pedestrian", "pedalcyclist"), .(
  f_ped_n = .N, f_ped_age = min(age, na.rm = TRUE),
  f_ped_injured = any(injured, na.rm = TRUE),
  f_ped_serious = any(susp_serious | killed, na.rm = TRUE)), by = Crash_ID]

CASES <- merge(cases, crash_cov, by = "Crash_ID", all.x = TRUE)
CASES <- merge(CASES, fem_drv, by = "Crash_ID", all.x = TRUE)
CASES <- merge(CASES, fem_ped, by = "Crash_ID", all.x = TRUE)
CASES[, has_f1549 := Crash_ID %in% crf$Crash_ID]
CASES[, role_matched := fifelse(preg_role_choice == "driver", !is.na(f_driver_n),
                        fifelse(preg_role_choice == "pedestrian_or_other", !is.na(f_ped_n), NA))]

write_fst(CASES, file.path(P2, "data/stageB/cases_covariates.fst"), compress = 70)
fwrite(CASES, file.path(P2, "data/stageB/cases_covariates.csv"))

cat("\n-- case linkage --\n")
cat(sprintf("cases with any female 15-49 person row: %s / %s (%.1f%%)\n",
            format(sum(CASES$has_f1549), big.mark = ","), format(nrow(CASES), big.mark = ","),
            100 * mean(CASES$has_f1549)))
cat(sprintf("Jev role=driver cases with a female 15-49 driver row: %s / %s (%.1f%%)\n",
            format(sum(CASES$preg_role_choice == "driver" & !is.na(CASES$f_driver_n)), big.mark = ","),
            format(sum(CASES$preg_role_choice == "driver"), big.mark = ","),
            100 * mean(!is.na(CASES$f_driver_n[CASES$preg_role_choice == "driver"]))))

# ---- 2. documentation-model frame ----------------------------------------------
case_driver_ids <- unique(CASES[preg_role_choice == "driver", Crash_ID])
all_drv <- fem[role == "driver"]
all_drv <- merge(all_drv, all_drv[, .(n_fdrv = .N), by = Crash_ID], by = "Crash_ID")
all_drv[, y_any := as.integer(Crash_ID %in% case_driver_ids)]

n_amb_case <- uniqueN(all_drv[y_any == 1L & n_fdrv > 1L, Crash_ID])
n_amb_pers <- sum(all_drv$y_any == 1L & all_drv$n_fdrv > 1L)
cat(sprintf("\nambiguous case crashes (>1 female 15-49 driver): %s crashes, %s persons -- dropped\n",
            format(n_amb_case, big.mark = ","), format(n_amb_pers, big.mark = ",")))

elig <- all_drv[n_fdrv == 1L]
elig[, y := y_any]
n_case <- sum(elig$y); n_ctrl_all <- sum(elig$y == 0L)
cat(sprintf("eligible female 15-49 SOLE drivers: %s   documented cases: %s   prevalence %.5f%%\n",
            format(nrow(elig), big.mark = ","), format(n_case, big.mark = ","),
            100 * n_case / nrow(elig)))

ctrl_idx <- sample.int(n_ctrl_all, size = min(NCTRL, n_ctrl_all))
FR <- rbind(elig[y == 1L], elig[y == 0L][ctrl_idx])
frac <- min(NCTRL, n_ctrl_all) / n_ctrl_all
FR[, samp_frac := fifelse(y == 1L, 1.0, frac)]
FR[, w := 1 / samp_frac]
FR <- merge(FR, crash_cov[, .(Crash_ID, Num_un, Num_pr, Pop_Group_ID, Onsys_Fl)],
            by = "Crash_ID", all.x = TRUE, suffixes = c("", ".dup"))
FR[, grep("\\.dup$", names(FR), value = TRUE) := NULL]

write_fst(FR, file.path(P2, "data/persons/docmodel_frame.fst"), compress = 70)
fwrite(FR, file.path(P2, "data/persons/docmodel_frame.csv"))

meta <- list(tau = TAU, seed = SEED,
             all_female_drivers_15_49 = nrow(all_drv),
             ambiguous_case_crashes_dropped = n_amb_case,
             ambiguous_case_persons_dropped = n_amb_pers,
             eligible_female_sole_drivers_15_49 = nrow(elig),
             documented_cases = n_case,
             population_prevalence = n_case / nrow(elig),
             controls_available = n_ctrl_all, controls_sampled = min(NCTRL, n_ctrl_all),
             control_sampling_fraction = frac,
             intercept_correction_logit = log(frac),
             frame_rows = nrow(FR),
             note = paste("Controls sampled at rate `control_sampling_fraction`; odds ratios",
                          "from an unweighted fit are unbiased, the intercept is shifted by",
                          "log(frac) and must be corrected before any predicted probability",
                          "is reported. Column `w` = 1/samp_frac for a weighted fit."))
writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
           file.path(P2, "data/persons/docmodel_meta.json"))
cat("\nDONE\n"); print(meta[c("documented_cases", "controls_sampled", "population_prevalence")])
