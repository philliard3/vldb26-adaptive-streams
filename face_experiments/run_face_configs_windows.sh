mkdir all_logs

for DUPLICATE_BOXES in {1..10}; do
    mkdir log target_debugging_outputs log_outputs
    echo "Running experiments with DUPLICATE_BOXES=$DUPLICATE_BOXES"
    for d in $(ls -d face_configs_std/*); do

        echo "deploying for folder $d"
        start_outer=$(date +%s)

        for f in $(cd $d ; ls *aq*); do
            num_experiment=$(pgrep -f face_query | wc -l)
            echo "cleaning up $num_experiment experiment processes"
            sudo pgrep -f face_query | xargs kill -9
            sleep 3
            echo "deploying $f"
            start_inner=$(date +%s)
            cp $d/$f tmp_time_series_config_deployment.json
            mkdir -p log_outputs
            output_file_name=target_debugging_outputs/tmp_output_${DUPLICATE_BOXES}_$d_$f.log

            AQUIFER_LARGE_EXCESS_PUNISHMENT=1 AQUIFER_BUDGET_CALCULATION_VERSION=2 OMZ_STRIDED_CONVERSION_METHOD_STATIC=34 OMZ_USE_NEAREST_RESIZE=1 REROUTE_ZERO_OPTION=0 WATERSHED_ONNX_USE_PINNED_MEMORY=0 WATERSHED_ONNX_INFERENCE_OLD=1 LOOP_ITEMS=1 BATCH_SIZE=1 DUPLICATE_BOXES=$DUPLICATE_BOXES NO_CROP=1 RUST_BACKTRACE=full time ./face_query tmp_time_series_config_deployment.json log_config.yml > $output_file_name 2>&1

            if [ $? -eq 0 ]; then
                rm $output_file_name
            fi
            mv log/output.log log/output_${DUPLICATE_BOXES}_$d_$f.log
            mv log_outputs log/compact_log_${DUPLICATE_BOXES}_$d_$f
            end_inner=$(date +%s)
            inner_duration=$((end_inner - start_inner))
            echo "Time taken for $f: $inner_duration seconds"
            sleep 3
        done

        end_outer=$(date +%s)
        outer_duration=$((end_outer - start_outer))
        echo "Total time taken for folder $d: $outer_duration seconds"
    done
    # Move all logs for this DUPLICATE_BOXES value into a parent folder
    mkdir -p all_logs/duplicate_boxes_${DUPLICATE_BOXES}
    mv log/ all_logs/duplicate_boxes_${DUPLICATE_BOXES}/log
    mv log_outputs/ all_logs/duplicate_boxes_${DUPLICATE_BOXES}/log_outputs
    mv target_debugging_outputs/ all_logs/duplicate_boxes_${DUPLICATE_BOXES}/target_debugging_outputs
    sleep 1
    mkdir log target_debugging_outputs log_outputs
    sleep 1

done
