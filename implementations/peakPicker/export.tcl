open_project proj_peakPicker
set_top peakPicker
add_files peakPicker.cpp
add_files -tb peakPicker_tb.cpp
add_files -tb ../../data/locations_3_ref.txt
add_files -tb ../../data/pssCorrMagSq_3_in.txt
add_files -tb ../../data/threshold_in.txt
open_solution solution1
set_part {xc7k410t-ffg900-2}
create_clock -period 3.90 -name default
set_clock_uncertainty 12.5%
export_design -format ip_catalog
exit
