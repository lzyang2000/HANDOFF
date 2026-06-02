#!/usr/bin/env bash
# Stability-trained variant of ckpt_setup_seed.sh.
#
# Points at the WBC + NoBV loco teachers trained with whole-body stability
# rewards added (com_in_support_polygon, capture_point_in_support_polygon,
# ankle_hip_step, linear & angular momentum change penalties).
#
# Train order before sourcing this file:
#   1. ./train_wbc_teacher_stable_seed.sh <GPU>     -> stable WBC teacher
#   2. ./train_loco_teacher_nobv_stable_seed.sh <GPU>  -> stable NoBV loco teacher
#   3. Fill in TODO_TIMESTAMP_* slots below with the run dir prefixes from
#      logs/rsl_rl/<exp>/<run>/ (the run dir's leading "<date>_<time>_<run-name>").
#   4. ./train_moe_student_unicmd_nobv_fullstable_seed.sh <GPU>  -> full-chain student
#
# This drives the "fully stable" experiment: student distills from teachers
# that themselves saw the stability rewards during training, so the KL target
# is consistent with the env reward shaping the student sees.

# Stable WBC teacher -- from train_wbc_teacher_stable_seed.sh.
export WBC_TEACHER_EXP="2026-06-01_02-50-43_g1_wbc_teacher_stable_seed"
export WBC_TEACHER_PROJ="logs/rsl_rl/g1_wbc_teacher_stable_seed"
export WBC_TEACHER_CKPT="-1"

# Stable NoBV loco teacher -- from train_loco_teacher_nobv_stable_seed.sh.
export LOCO_TEACHER_NOBV_EXP="2026-06-01_02-50-52_g1_loco_teacher_nobv_stable_seed"
export LOCO_TEACHER_NOBV_PROJ="logs/rsl_rl/g1_loco_teacher_nobv_stable_seed"
export LOCO_TEACHER_NOBV_CKPT="-1"

# Stable privileged loco teacher -- not yet wired up (no -Stable env_cfg
# exists for the privileged Loco-Teacher-Flat-Unitree-G1 task). Left as a
# placeholder so any downstream script reading LOCO_TEACHER_* still resolves.
export LOCO_TEACHER_EXP="TODO_TIMESTAMP_g1_loco_teacher_stable_seed"
export LOCO_TEACHER_PROJ="logs/rsl_rl/g1_loco_teacher_stable_seed"
export LOCO_TEACHER_CKPT="-1"
