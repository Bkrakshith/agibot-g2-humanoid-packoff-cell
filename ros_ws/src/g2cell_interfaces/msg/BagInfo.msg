# One salad bag on the cell (spawned by the Stage 5 scene controller; Stage 2 has 3 rigid placeholders).
uint8 VARIANT_SMALL=0
uint8 VARIANT_STANDARD=1
uint8 VARIANT_LARGE=2

# Sort label (spec "Sort rule"): "good" goes to the pack crate, "reject" and any other SKU name
# go to their rack lane.
string LABEL_GOOD=good
string LABEL_REJECT=reject

string id                        # e.g. "Bag_01"
uint8 variant
string label                     # "good" | "reject" | <SKU name>
geometry_msgs/PoseStamped pose   # frame "cell" = cell frame (origin = G2 base on the floor)
