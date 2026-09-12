"""
Tests for the frozen-particles feature (frozen_particles_file / frozen_strands input options).

System: an 8-bp oxRNA2 duplex (16 nucleotides). frozen.txt freezes particles 0-3 and 8-11,
input_vmmc freezes strand 0 (particles 0-7; strand ids are 0-based) instead. The tests are

  a) exact preservation: positions and orientations of frozen particles are bit-identical after
     1e6 MC moves (MC2 with translation, rotation, VMMC and pivot moves; MC; VMMC) and 1e5 MD steps,
     while movable particles do move;
  b) energy consistency: (i) the VMMC backend checks its tallied energy against a full
     recomputation at every step and throws on a mismatch; (ii) for random single-particle trial
     moves with frozen neighbours, the local energy difference (the quantity the MC moves use)
     equals the full-system energy difference;
  c) reference-sampler agreement: the C++ MC2 sampler with frozen particles and an independent
     Python Metropolis sampler over the movable particles (using oxpy energy calls) give the same
     mean potential energy and interface distance within statistical error;
  d) collective-move safety: covered by (a), whose MC2 run includes VMMC and pivot moves that are
     actually attempted (checked from the acceptance columns of the energy file).

Results are written to frozen_results.dat (one PASS/FAIL line per check) and compared with
frozen_results_correct.dat by the test suite.
"""
import math
import sys

import numpy as np
import oxpy

results = []


def record(name, ok, detail=""):
    results.append("%s %s" % (name, "PASS" if ok else "FAIL"))
    print("%-45s %s %s" % (name, "PASS" if ok else "FAIL", detail))


def snapshot(particles):
    return {p.index: (p.pos.copy(), p.orientation.copy(), p.orientationT.copy(), p.vel.copy(), p.L.copy()) for p in particles}


def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    x, y, z = axis
    return np.array([[x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                     [x * y * C + z * s, y * y * C + c, y * z * C - x * s],
                     [x * z * C - y * s, y * z * C + x * s, z * z * C + c]])


def run_and_check(name, input_name, overrides, steps, expect_frozen):
    """Runs a backend for `steps` steps and checks that frozen particles are bit-identical."""
    with oxpy.Context(print_coda=False):
        inp = oxpy.InputFile()
        inp.init_from_filename(input_name)
        for k, v in overrides.items():
            inp[k] = v
        manager = oxpy.OxpyManager(inp)
        ci = manager.config_info()
        particles = ci.particles()
        frozen = sorted(p.index for p in particles if p.frozen)
        movable = sorted(p.index for p in particles if not p.frozen)
        record(name + "_flags", frozen == expect_frozen and list(ci.movable_particles) == movable, str(frozen))
        before = snapshot(particles)
        manager.run(steps, print_output=True)
        after = snapshot(particles)
        frozen_ok = all(np.array_equal(before[i][k], after[i][k]) for i in frozen for k in range(3))
        record(name + "_frozen_bit_identical", frozen_ok)
        moved = all((not np.array_equal(before[i][0], after[i][0])) for i in movable)
        record(name + "_movable_moved", moved)
        return particles


# ---------------------------------------------------------------- (a) + (d): MC2 with all move types
N_MOVES = 1000000
run_and_check("a_MC2", "input_mc2", {"steps": str(N_MOVES // 16), "print_energy_every": "1000", "log_file": "log_a_mc2.dat"},
              N_MOVES // 16, [0, 1, 2, 3, 8, 9, 10, 11])
# acceptance ratios of the four moves are printed at the beginning of each energy line
last = open("energy_mc2.dat").readlines()[-1].split()
acc = [float(x) for x in last[2:6]]
record("d_MC2_vmmc_and_pivot_attempted_and_accepted", acc[2] > 0.0 and acc[3] > 0.0, "acceptances %s" % acc)

# ---------------------------------------------------------------- (a): MC backend
run_and_check("a_MC", "input_mc", {"steps": str(N_MOVES // 16), "print_energy_every": "1000", "log_file": "log_a_mc.dat"},
              N_MOVES // 16, [0, 1, 2, 3, 8, 9, 10, 11])

# ---------------------------------------------------------------- (a) + (b-i): VMMC backend, frozen strand, energy check every step
try:
    run_and_check("a_VMMC", "input_vmmc", {"steps": str(N_MOVES // 16), "check_energy_every": "1", "print_energy_every": "1000", "log_file": "log_a_vmmc.dat"},
                  N_MOVES // 16, list(range(8)))
    record("b_VMMC_tallied_energy_matches_recomputed", True)
except Exception as e:
    record("b_VMMC_tallied_energy_matches_recomputed", False, str(e))

# ---------------------------------------------------------------- (a): MD
N_MD = 100000
particles = run_and_check("a_MD", "input_md", {"steps": str(N_MD), "print_energy_every": "10000", "log_file": "log_a_md.dat"}, N_MD, [0, 1, 2, 3, 8, 9, 10, 11])

# ---------------------------------------------------------------- (b-ii): local vs full energy differences for trial moves with frozen neighbours
with oxpy.Context(print_coda=False):
    inp = oxpy.InputFile()
    inp.init_from_filename("input_mc")
    inp["list_type"] = "no"
    inp["log_file"] = "log_b.dat"
    manager = oxpy.OxpyManager(inp)
    ci = manager.config_info()
    particles = ci.particles()
    inter = ci.interaction
    movable = [p for p in particles if not p.frozen]

    def local_energy(p):
        e = 0.0
        for q in (p.n3, p.n5):
            if q is not None:
                e += inter.pair_interaction_bonded(p, q)
        for q in particles:
            if q.index != p.index:
                e += inter.pair_interaction_nonbonded(p, q)
        return e

    rng = np.random.default_rng(1234)
    max_dev = 0.0
    n_finite = 0
    for trial in range(400):
        p = movable[rng.integers(len(movable))]
        E0_full = manager.system_energy()
        E0_loc = local_energy(p)
        pos0, or0 = p.pos.copy(), p.orientation.copy()
        if trial % 2 == 0:
            p.pos = pos0 + rng.uniform(-0.1, 0.1, 3)
        else:
            R = rotation_matrix(rng.normal(size=3), rng.uniform(0.0, 0.3))
            p.orientation = or0 @ R
            p.orientationT = p.orientation.T.copy()
            p.set_positions()
        E1_full = manager.system_energy()
        E1_loc = local_energy(p)
        if math.isfinite(E1_full) and abs(E1_full) < 1e10:
            n_finite += 1
            max_dev = max(max_dev, abs((E1_loc - E0_loc) - (E1_full - E0_full)))
        p.pos = pos0
        p.orientation = or0
        p.orientationT = or0.T.copy()
        p.set_positions()
    record("b_local_delta_equals_full_delta", n_finite > 300 and max_dev < 1e-9, "max deviation %.2e over %d finite trials" % (max_dev, n_finite))
    del manager

# ---------------------------------------------------------------- (c): reference sampler agreement
DELTA_T, DELTA_R = 0.1, 0.2
N_SWEEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
SAMPLE_EVERY = 10


def min_image_distance(a, b, box_sides):
    d = a - b
    d -= box_sides * np.rint(d / box_sides)
    return float(np.linalg.norm(d))


def observables(particles):
    U = manager.system_energy()
    L = manager.config_info().box_sides
    d_int = min_image_distance(particles[7].pos, particles[8].pos, L)  # interface pair (movable 7, frozen 8)
    d_end = min_image_distance(particles[15].pos, particles[0].pos, L)  # movable 15 opposite frozen 0
    return U, d_int, d_end


def block_stats(x, n_blocks=20):
    x = np.asarray(x)
    x = x[:len(x) // n_blocks * n_blocks].reshape(n_blocks, -1).mean(axis=1)
    return x.mean(), x.std(ddof=1) / math.sqrt(n_blocks)


# C++ sampler: MC2 with translation + rotation moves only, frozen particles
cpp = []
with oxpy.Context(print_coda=False):
    inp = oxpy.InputFile()
    inp.init_from_filename("input_mc2_tr")
    inp["seed"] = "4242"
    inp["log_file"] = "log_c_cpp.dat"
    inp["energy_file"] = "energy_c_cpp.dat"
    inp["trajectory_file"] = "trajectory_c_cpp.dat"
    inp["steps"] = str(N_SWEEPS)
    inp["print_energy_every"] = str(N_SWEEPS)
    manager = oxpy.OxpyManager(inp)
    particles = manager.config_info().particles()
    manager.run(N_SWEEPS // 5, print_output=False)  # equilibration
    for i in range(N_SWEEPS // SAMPLE_EVERY):
        manager.run(SAMPLE_EVERY, print_output=False)
        cpp.append(observables(particles))
    del manager  # a manager destroyed after the next one is created would clear the shared ConfigInfo

# Python reference sampler: Metropolis over the movable particles only, using oxpy energy calls
py = []
with oxpy.Context(print_coda=False):
    inp = oxpy.InputFile()
    inp.init_from_filename("input_mc")
    inp["list_type"] = "no"
    inp["log_file"] = "log_c_py.dat"
    manager = oxpy.OxpyManager(inp)
    ci = manager.config_info()
    particles = ci.particles()
    T = ci.temperature
    movable = [p for p in particles if not p.frozen]
    rng = np.random.default_rng(777)
    E = manager.system_energy()
    n_acc = 0
    n_tot = 0
    for sweep in range(N_SWEEPS + N_SWEEPS // 5):
        for k in range(2 * len(movable)):
            p = movable[rng.integers(len(movable))]
            pos0, or0, orT0 = p.pos.copy(), p.orientation.copy(), p.orientationT.copy()
            if rng.random() < 0.5:
                p.pos = pos0 + rng.uniform(-DELTA_T, DELTA_T, 3)
            else:
                R = rotation_matrix(rng.normal(size=3), rng.uniform(0.0, DELTA_R))
                p.orientation = or0 @ R
                p.orientationT = p.orientation.T.copy()
                p.set_positions()
            E_new = manager.system_energy()
            n_tot += 1
            if E_new < E or math.exp(-(E_new - E) / T) > rng.random():
                E = E_new
                n_acc += 1
            else:
                p.pos = pos0
                p.orientation = or0
                p.orientationT = orT0
                p.set_positions()
        if sweep >= N_SWEEPS // 5 and (sweep - N_SWEEPS // 5) % SAMPLE_EVERY == 0:
            py.append(observables(particles))
    frozen_ok = all(bool(p.frozen) for p in particles if p.index in (0, 1, 2, 3, 8, 9, 10, 11))
    print("python sampler acceptance %.3f" % (n_acc / n_tot))
    del manager

cpp = np.array(cpp)
py = np.array(py)
for j, label in enumerate(("U", "d_interface", "d_end")):
    m1, s1 = block_stats(cpp[:, j])
    m2, s2 = block_stats(py[:, j])
    z = abs(m1 - m2) / math.sqrt(s1 ** 2 + s2 ** 2)
    record("c_reference_sampler_%s" % label, z < 4.0, "cpp %.4f+-%.4f py %.4f+-%.4f z=%.2f" % (m1, s1, m2, s2, z))

# ---------------------------------------------------------------- input validation
with oxpy.Context(print_coda=False):
    inp = oxpy.InputFile()
    inp.init_from_filename("input_mc2")
    inp["fix_diffusion"] = "true"
    inp["log_file"] = "log_val.dat"
    try:
        manager = oxpy.OxpyManager(inp)
        record("validation_fix_diffusion_rejected", False)
    except Exception as e:
        record("validation_fix_diffusion_rejected", "fix_diffusion" in str(e))

# ---------------------------------------------------------------- frozen_skip_bonded_pairs
# distorted configuration: frozen particle 1 displaced by 0.6 length units, which puts the frozen-frozen
# backbone bonds 0-1 and 1-2 far outside the FENE range
with open("duplex.dat") as f:
    lines = f.read().splitlines()
fields = lines[4].split()
fields[0] = repr(float(fields[0]) + 0.6)
lines[4] = " ".join(fields)
with open("duplex_distorted.dat", "w") as f:
    f.write("\n".join(lines) + "\n")


def energy_with(conf, skip, run_steps=0):
    with oxpy.Context(print_coda=False):
        inp = oxpy.InputFile()
        inp.init_from_filename("input_mc2")
        inp["conf_file"] = conf
        inp["frozen_skip_bonded_pairs"] = "true" if skip else "false"
        inp["log_file"] = "log_skip.dat"
        inp["energy_file"] = "energy_skip.dat"
        inp["trajectory_file"] = "trajectory_skip.dat"
        manager = oxpy.OxpyManager(inp)
        ci = manager.config_info()
        particles = ci.particles()
        E = manager.system_energy()
        frozen_bonded = 0.0
        for p in particles:
            if p.frozen and p.n3 is not None and p.n3.frozen:
                frozen_bonded += ci.interaction.pair_interaction_bonded(p, p.n3)
        before = snapshot(particles)
        moved = None
        if run_steps > 0:
            manager.run(run_steps, print_output=False)
            after = snapshot(particles)
            frozen_ok = all(np.array_equal(before[p.index][k], after[p.index][k]) for p in particles if p.frozen for k in range(3))
            moved = frozen_ok and all(not np.array_equal(before[p.index][0], after[p.index][0]) for p in particles if not p.frozen)
        E_after = manager.system_energy()
        del manager
    return E, frozen_bonded, moved, E_after

E_full, fb, _, _ = energy_with("duplex.dat", False)
E_skip, fb2, _, _ = energy_with("duplex.dat", True)
record("skip_bonded_energy_identity", abs((E_full - fb) - E_skip) < 1e-9 * max(1.0, abs(E_full)), "E_full=%.6f frozen-frozen bonded=%.6f E_skip=%.6f" % (E_full, fb, E_skip))
try:
    E_dist_full, fb_d, _, _ = energy_with("duplex_distorted.dat", False)
    record("skip_bonded_distorted_without_option_refused_or_huge", E_dist_full > 1e11, "E=%.3e" % E_dist_full)
except Exception as e:
    record("skip_bonded_distorted_without_option_refused_or_huge", "bonded neighbors" in str(e), str(e))
E_dist_skip, _, moved_ok, E_dist_after = energy_with("duplex_distorted.dat", True, run_steps=100000 // 16)
record("skip_bonded_distorted_finite_and_frozen_preserved", abs(E_dist_skip) < 1e3 and abs(E_dist_after) < 1e3 and moved_ok, "E=%.4f -> %.4f" % (E_dist_skip, E_dist_after))

with open("frozen_results.dat", "w") as f:
    f.write("\n".join(results) + "\n")
