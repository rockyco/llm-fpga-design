open_project proj_peakPicker
set_top peakPicker
add_files peakPicker.cpp
add_files -tb peakPicker_tb.cpp
add_files -tb ../../data/locations_3_ref.txt
add_files -tb ../../data/pssCorrMagSq_3_in.txt
add_files -tb ../../data/threshold_in.txt
open_solution solution1
set_part {xc7k410t-ffg900-2}
csim_design
exit
