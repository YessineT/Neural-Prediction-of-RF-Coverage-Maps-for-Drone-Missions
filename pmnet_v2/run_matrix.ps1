# Experiment matrix, PMNet with Yessine's setup.  Run from the folder with the .py files:   .\run_matrix.ps1
# 1) Put the options of YOUR small PMNet in $model (the first line printed by a run shows its parameter count).
# 2) A finished run (results.json exists) is skipped, an interrupted run resumes: you can stop and restart the script.
# 3) Comment out ("#") the stages you do not want to run yet.
$model = @("--width", "0.22")

function Run($name, $extra) {
    if (Test-Path "runs/$name/results.json") { Write-Host "skip $name (done)"; return }
    python train_yessine_setup.py @model @extra --out "runs/$name"
}

# Stage 1, seed 0, 7 input channels: the four losses of his study 1
Run "pm_mse_nolos_s0"   @("--loss", "mse")
Run "pm_l1_nolos_s0"    @("--loss", "l1")
Run "pm_huber_nolos_s0" @("--loss", "huber")
Run "pm_combo_nolos_s0" @("--loss", "combo")

# Stage 2, seed 0, with the line-of-sight channel, plus the asymmetric no-signal loss
Run "pm_mse_los_s0"        @("--loss", "mse", "--los")
Run "pm_combo_los_s0"      @("--loss", "combo", "--los")
Run "pm_combo_los_asym_s0" @("--loss", "combo", "--los", "--floor-method", "asymmetric")

# Stage 3, seeds 1 and 2 for the key configurations (baseline, combo, LoS, asymmetric), as the supervisor asked
foreach ($s in 1, 2) {
    Run "pm_mse_nolos_s$s"        @("--loss", "mse", "--seed", "$s")
    Run "pm_combo_nolos_s$s"      @("--loss", "combo", "--seed", "$s")
    Run "pm_combo_los_s$s"        @("--loss", "combo", "--los", "--seed", "$s")
    Run "pm_combo_los_asym_s$s"   @("--loss", "combo", "--los", "--floor-method", "asymmetric", "--seed", "$s")
}

# Discrete visibility maps for the deliverable (independent of the training runs)
# python compute_los.py --out los_cache --mask-valid --check