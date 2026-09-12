# Frozen particles

Frozen particles are particles whose position and orientation are never changed by the simulation, while they keep interacting with every other particle. They provide a rigid boundary for the movable particles: the typical use is to sample the flexible parts of a large structure (for instance unresolved loops of a cryo-EM model) with the resolved part held exactly at its reference coordinates.

## Usage

```text
frozen_particles_file = frozen.txt   # indices or inclusive ranges, one or more per line, e.g. "0-9 15 20-24"
frozen_strands = 0, 3                # optional: freeze whole strands (0-based internal strand ids)
fix_diffusion = false                # mandatory when particles are frozen
```

Frozen particles are supported by the CPU `MD`, `MC`, `MC2`, `VMMC` and `PT_VMMC` backends. The CUDA backend, the minimisation backends and volume-changing moves (`volume`, `molecule_volume`, `shape` in `MC2`, the `NPT` ensemble in `MC`, the MD barostat) raise an error. `fix_diffusion` re-centres whole molecules and must be disabled: the code refuses to start otherwise.

From `oxpy`, the flag is available as `BaseParticle.frozen`, and `ConfigInfo.has_frozen_particles` / `ConfigInfo.movable_particles` expose the movable set.

### Frozen structures that are not at the model's minimum

A frozen set is typically taken from an experimental structure, whose backbone geometry is not exactly that of the coarse-grained model: some backbone bonds between frozen particles fall outside the FENE range, where oxDNA/oxRNA return an energy of {math}`10^{12}`. Bonded energies between two frozen particles are constants that cannot affect the distribution of the movable particles, so the option

```text
frozen_skip_bonded_pairs = true
```

makes the full-system energy computations (`get_system_energy`, MD forces, the `MC` backend's energy tally, the initial topology sanity check) skip every bonded pair whose two members are frozen. Nonbonded pairs are still computed, and pairs involving at least one movable particle are always computed. Interactions between a movable particle and a frozen one are physical and must be sane: the bonds at the junctions between a frozen segment and a flexible one have to be inside the FENE range before the simulation starts. The single-particle MC moves evaluate only pairs involving the moved (hence movable) particle, so their acceptance is unaffected by the option; what changes is only the constant offset of the reported total energy. The option is refused by the `VMMC` and `PT_VMMC` backends, whose energy bookkeeping is not covered.

## What the samplers do

The frozen set is read once by `SimBackend::_init_frozen_particles()`, which sets `BaseParticle::frozen` and caches the list of movable indices in `ConfigInfo::movable_particles`.

* **Single-particle moves** (`MCTras`, `MCRot`, `RotateSite` in `MC2`; translations and rotations in `MC`) draw the particle uniformly from the movable list instead of from all particles. Drawing from all particles and rejecting frozen ones would be equally correct but would waste most attempts when most of the system is frozen. Because the proposal is uniform over the movable set both for the forward and for the reverse move, the proposal is symmetric and the usual Metropolis acceptance is unchanged.
* **Pivot** (`MC2`): the pivot particle is drawn from all particles, the set of particles downstream of the pivot is enumerated, and the move is rejected before any energy computation if any member of that set is frozen. Note that this move rotates *everything* between the pivot and the end of the strand, so it is only useful for flexible segments that reach a strand end; an internal flexible segment flanked by frozen particles on both sides always contains a frozen particle downstream and the move is always rejected there.
* **VMMC** (`MC2` move and `VMMC`/`PT_VMMC` backends): the seed is drawn uniformly from the movable particles; if the cluster construction would recruit a frozen particle the move is rejected. In the `MC2` move the rejection is immediate; in the `VMMC` backend the frozen particle is left in the "prelinked" state, and the pre-existing rule that rejects any move with a prelinked particle outside the final cluster takes care of the rejection (this keeps the energy bookkeeping of that backend untouched). Cluster moves whose cluster does not touch a frozen particle are unaffected.
* **MD** (`MD_CPUBackend`): frozen particles are skipped in both halves of the velocity-Verlet integrator, their velocities and angular momenta are set to zero at initialisation (`refresh_vel` or not) and again after every thermostat call, and they are excluded from the centre-of-mass momentum reset. Forces on them are still computed (they are needed for the movable particles) but never applied.

## Detailed balance

For the single-particle moves the argument is the standard one: the proposal density is symmetric (uniform choice among the movable particles, symmetric displacement or rotation) and the Metropolis acceptance uses the full energy change, including the interactions with frozen neighbours.

For the cluster moves the relevant property is *superdetailed balance*, which VMMC satisfies cluster by cluster (Whitelam and Geissler, J. Chem. Phys. 127, 154101 (2007); Whitelam et al., Soft Matter 5, 1251 (2009)): for every cluster $C$ and displacement map $\mathcal{M}$, the flux from configuration $\mu$ to $\nu = \mathcal{M}(\mu)$ through $C$ equals the flux from $\nu$ to $\mu$ through $C$ with the inverse map. oxDNA's implementation realises this with the usual link-formation probabilities $p_{ij} = \max(0, 1 - e^{-\beta(\epsilon_{ij}(\nu) - \epsilon_{ij}(\mu))})$, the "prelink" trick that samples the reverse-move factor, and two early rejections: clusters larger than `max_cluster_size`, and clusters containing a prelinked-but-not-recruited (frustrated) particle.

Rejecting a move because the cluster contains a frozen particle is an additional criterion of the same kind as the size limit: it is a function of the cluster $C$ alone, $I(C) = 1$ if $C$ contains no frozen particle and $0$ otherwise. The reverse move from $\nu$ through the same cluster $C$ has exactly the same value of $I(C)$, so multiplying both the forward and the reverse acceptance by $I(C)$ leaves the ratio of the two fluxes unchanged and superdetailed balance holds for every cluster. Two details make the implementation match this argument:

1. Recruitment is permanent during cluster growth, so rejecting as soon as a frozen particle is recruited (instead of finishing the cluster and then rejecting) gives the same outcome: $I(C) = 0$ for every cluster that would have been completed.
2. The seed must be drawable with the same probability in both directions. It is drawn uniformly from the movable particles, and every particle of an accepted cluster is movable (by $I(C) = 1$), so the reverse move can choose the same seed with the same probability $1/N_\mathrm{movable}$.

Frozen particles that are tested as link candidates but *not* recruited are ordinary boundary particles: their link probabilities enter the acceptance exactly as those of any other non-recruited neighbour, and nothing is changed there. The `VMMC` backend variant (leaving a would-be-recruited frozen particle prelinked) is equivalent because a prelinked particle outside the final cluster already causes rejection.

The Pivot move is a deterministic symmetric proposal (random pivot, random direction, random rotation with symmetric angle distribution); the frozen criterion depends only on the moved set, which is identical for the reverse move.

## Tests

`test/FROZEN` (run with the `oxpy` level of the test suite, `python TestSuite.py test_folder_list.txt python oxpy` from the `test` folder) checks on an 8-bp oxRNA2 duplex with half of the nucleotides frozen:

* exact preservation: positions and orientations of frozen particles are bit-identical after $10^6$ MC moves (`MC2` with translation, rotation, VMMC and pivot moves; `MC`; `VMMC`) and $10^5$ MD steps, while the movable particles do move; the VMMC and pivot moves are verified to be attempted and accepted;
* energy consistency: the `VMMC` backend compares its tallied energy with a full recomputation at every step and throws on any mismatch; independently, for 400 random single-particle trial moves with frozen neighbours the local energy difference used by the moves equals the full-system energy difference to $10^{-13}$;
* reference-sampler agreement: the `MC2` sampler with frozen particles and an independent Metropolis sampler written in Python over the movable particles (using `oxpy` energy calls only) agree on the mean potential energy and on two frozen-movable distances within statistical error (block-averaged, $z < 4$). Run `python oxpy_input.py 60000` for a ten-times longer comparison;
* the input file is rejected if `fix_diffusion` is left enabled;
* `frozen_skip_bonded_pairs`: on the undistorted duplex the skipped energy equals exactly the sum of the bonded energies of the frozen-frozen pairs; with one frozen particle displaced so that its two frozen-frozen backbone bonds are far outside the FENE range, the run starts, its energy stays finite, the frozen particles stay bit-identical and the movable particles keep moving; without the option the same input reports an energy of the order of {math}`10^{12}`.

Acceptance rates are deliberately *not* compared with an unfrozen run: freezing changes the environment that is sampled, so the rates legitimately differ.
