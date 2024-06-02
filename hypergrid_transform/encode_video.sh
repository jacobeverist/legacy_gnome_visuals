#!/usr/bin/env bash


# convert png files to video format

#ffmpeg -i  out/floating_illustration_%05d.png out/out.mkv
#ffmpeg -y -i  out/hypergrid_2D_frames_1_%05d.png out/out.mkv

#ffmpeg -y -i  out/hypergrid_2D_frames_1_%05d.png -c:v libx264 -preset ultrafast -crf 0 out/out.mkv
#ffmpeg -y -i  out/hypergrid_2D_frames_1_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out.mkv
#ffmpeg -y -i  out/hypergrid_2D_frames_3_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out.mkv
#ffmpeg -y -i  out/hypergrid_2D_frames_4_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out.mkv
#ffmpeg -y -i  out/hypergrid_2D_frames_5_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out.mkv
#ffmpeg -y -i  out/hypergrid_2D_similarity_frames_1_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out.mkv
#ffmpeg -y -i  out/hypergrid_2D_similarity_frames_2_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out.mkv


#ffmpeg -y -i out/grid_2D_0_90_angles_1.0_periods_4_bins_v1_frames_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out/grid_2D_0_90_angles_1.0_periods_4_bins_v1.mkv
#ffmpeg -y -i out/grid_2D_0_90_angles_1.0_periods_4_bins_v1_.mkv -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" out/grid_2D_0_90_angles_1.0_periods_4_bins_v1_.gif

#ffmpeg -y -i out/grid_2D_0_90_angles_1.0_periods_4_bins_v1_frames_%05d.png -c:v libx264 -preset veryslow -crf 0 out/out/grid_2D_0_90_angles_1.0_periods_4_bins_v1.mkv
#ffmpeg -y -i out/out/grid_2D_0_90_angles_1.0_periods_4_bins_v1_.mkv -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" out/out/grid_2D_0_90_angles_1.0_periods_4_bins_v1_.gif

# convert video to gif
#ffmpeg -y -i out/out.mkv -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" out/out.gif

#declare -a filelist=("out/grid_2D_2_angles_1_periods_4_bins" "out/grid_2D_2_angles_2_periods_4_bins" "out/grid_2D_2_angles_3_periods_4_bins" "out/grid_2D_2_angles_1_periods_8_bins" "out/grid_2D_2_angles_2_periods_8_bins" "out/grid_2D_2_angles_3_periods_8_bins" "out/grid_2D_4_angles_1_periods_4_bins" "out/grid_2D_4_angles_2_periods_4_bins" "out/grid_2D_4_angles_3_periods_4_bins" "out/grid_2D_4_angles_1_periods_8_bins" "out/grid_2D_4_angles_2_periods_8_bins" "out/grid_2D_4_angles_3_periods_8_bins")
declare -a filelist=("out/visual_test")

for i in "${filelist[@]}"
do
    ffmpeg -y -framerate 10 -i "${i}_frames_1_%05d.png"  -c:v libx264 -preset veryslow -crf 0 "${i}.mkv"
    ffmpeg -y -i "${i}.mkv" -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" "${i}.gif"
done

#declare -a arr=("element1" "element2" "element3")




#export FILEROOT="out/grid_2D_0_90_angles_1.0_periods_8_bins_v1"
#ffmpeg -y -i "${FILEROOT}_frames_%05d.png"  -c:v libx264 -preset veryslow -crf 0 "${FILEROOT}.mkv"
#ffmpeg -y -i "${FILEROOT}.mkv" -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" "${FILEROOT}.gif"
