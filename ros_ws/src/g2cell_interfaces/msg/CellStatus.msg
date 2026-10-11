# Cell state published by g2cell_tasks/cell_manager (latched-style, ~2 Hz) on /g2cell/cell_status.
builtin_interfaces/Time stamp
string episode_id                # "" when no episode was started yet
bool episode_active
bool line_running                # conveyor running (Stage 3; always false until then)
bool crate_at_pack_position      # a crate is held by the stop gate at the pack position
uint16 crate_fill_count          # bags verified inside the pack-position crate
uint16 crate_fill_target         # bags per crate before release (spec: e.g. 10, set per episode)
uint16 bags_on_table
uint16 picks_done                # pick attempts finished this episode (any outcome)
string[] not_implemented         # parts of the status that are tracked/assumed, not sensed yet
