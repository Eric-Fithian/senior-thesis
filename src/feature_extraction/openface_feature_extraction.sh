# Docker command to run OpenFace Feature Extraction
docker run -it -v $(pwd)/data/interim/video:/data -v $(pwd)/data/processed:/output algebr/openface:latest

# Then inside the container run the following command for each video. Remember to replace {###} and {P/I} with the corresponding video file name.
build/bin/FeatureExtraction -f /data/P{###}_{P/I}.mp4 -out_dir /output -2Dfp -3Dfp -pdmparams -pose -aus -gaze