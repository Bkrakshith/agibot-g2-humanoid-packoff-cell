# Outcome of one pick-and-place, published on /g2cell/pick_results and returned by PickBag.
# Spec "per-pick outcomes to log" (codes 0-4):
uint8 OUTCOME_SUCCESS=0          # bag fully inside the correct crate or rack tote, at rest
uint8 OUTCOME_WRONG_TARGET=1     # bag in the wrong crate or lane
uint8 OUTCOME_DROP=2             # bag touches the floor or the conveyor outside a crate
uint8 OUTCOME_REGRASP=3          # placed correctly, but needed more than one grasp attempt
uint8 OUTCOME_OVERFLOW=4         # bag sitting proud of the crate rim
# Not spec outcomes: the task did not get as far as a scored placement.
uint8 OUTCOME_UNVERIFIED=10      # motion finished, but no placement observation (Stage 5 scorer missing)
uint8 OUTCOME_PLAN_FAILED=11     # IK / head clearance / motion failure before the bag was placed
uint8 OUTCOME_GRASP_FAILED=12    # every grasp attempt missed; the bag stayed on the table
uint8 OUTCOME_UNREACHABLE=13     # no arm can reach the bag or target (reachability table)
uint8 OUTCOME_ABORTED=14         # canceled or the motion server went away
uint8 OUTCOME_INVALID=15         # bad goal (unknown bag, unknown lane, ...)

builtin_interfaces/Time stamp
string bag_id
string arm                       # "left" | "right"
string intended_target           # "crate" | "rack_lane"
string intended_lane             # rack lane id (e.g. "L1_1") when intended_target = rack_lane
string actual_target             # as observed by the scorer; "" if not observed
string actual_lane
uint8 outcome
string outcome_name              # "success", "wrong_target", ...
bool verified                    # true only when a placement observation backed the outcome
uint8 grasp_attempts
string failed_stage              # "" or approach|grasp|lift|transfer|place|retract
float32 duration                 # s
string message
