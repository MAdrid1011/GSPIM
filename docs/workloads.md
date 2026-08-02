# Workload Adapters

Each adapter implements the same typed `ModelLayout` contract: PPIM program
metadata, aligned `(die, bank, block, slot)` source blocks, record expansion,
exact payload size, and per-frame 3D Gaussian generation. An adapter may not
add a bypass around PPIM, HCF, or the runtime.

These are mechanism adapters over the public fixture schema, not checkpoint
loaders or claims of bit-for-bit upstream workload reproduction. The GSPIM
paper categorizes the three temporal representations but does not define each
upstream checkpoint serialization. An upstream export is accepted only after it
is normalized into this explicit schema.

## Explicit 4D / 4DGS-1K

Each temporal record includes a spatial mean, temporal mean, 4D rotation, 4D
scale, and the aligned Gaussian payload. S2 constructs `R diag(s^2) R^T` and
conditions that 4D Gaussian on each timestamp to produce its 3D mean and
covariance. TEMP_ACTIVITY obtains the same temporal variance for the window
test. HCF compacts only records whose result is active; there is no keyframe-mask
OR path.

## Ex4DGS

Static records are active. Dynamic records are active only when their declared
closed keyframe support intersects the current half-open window: a support that
starts at the window end is excluded, while one ending at the window start is
included. Every
normalized record supplies its aligned physical location. S2
interpolates each normalized keyframe's position, covariance, opacity, and
color. A normalized export must supply every modeled field at every keyframe;
the adapter rejects legacy record-level rendering fields. Because S2 generates
every selected primitive for every frame in an active window, every selected
dynamic record must bracket each requested frame timestamp. This is checked
before S1 task submission or HCF allocation; an intersecting support interval
alone is not enough to admit an incomplete window.

## Anchored 4DGS

PPIM tests an anchor interval against the temporal window. HCF expands the
selected anchor's declared `binding_start` plus `binding_count` range. Every
bound ID must resolve to explicit representation-generated geometry for every
requested S2 timestamp, covariance, appearance, byte size, and an aligned
location in the anchor's die; malformed, missing, or cross-die ranges fail
before S1 task submission or HCF allocation. The paper specifies anchor selection and binding expansion, not a
universal bound-Gaussian motion equation, so this artifact never infers one
from a `velocity` field.

## External Inputs

The default CLI runs checked-in fixtures only. `scripts/fetch_workloads.py`
may create an ignored Ex4DGS checkout at its pinned revision, but it neither
vendors source nor guesses a checkpoint schema. A user-provided normalized
export is validated before it enters the common model.
