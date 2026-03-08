# deploy all time series files in each folder
# list of all folders that start with time_series_deployment_configs_*
for d in $(ls -d ./ecg_preclassifier_window_configs_may29_10pm/*); do
    # deploy all time series files in each folder
    echo "deploying for folder $d"
    start_outer=$(date +%s)
    for f in $(cd $d ; ls * ); do
        num_python=$(pgrep -f python | wc -l)
        echo "cleaning up $num_python python processes"
        sudo pgrep -f python | xargs kill -9
        sleep 5
        num_experiment=$(pgrep -f time_series | wc -l)
        echo "cleaning up $num_experiment time_series processes"
        sudo pgrep -f time_series | xargs kill -9
        sleep 5
        echo "deploying $f"
        start_inner=$(date +%s)
        # echo "my test output" > my_test_output.txt
        cp $d/$f tmp_time_series_config_deployment.json
        # make a combined file name for the output
        mkdir log_outputs
        output_file_name=target_debugging_outputs/tmp_output_$d_$f.log
        ORT_DYLIB_PATH=$ORT_DYLIB_PATH time ./time_series_query tmp_time_series_config_deployment.json log_config.yml > $output_file_name 2>&1
        if [ $? -eq 0 ]; then
            rm $output_file_name
        fi
        mv log/output.log log/output_$d_$f.log
        mv log_outputs log/compact_log_$d_$f
        end_inner=$(date +%s)
        inner_duration=$((end_inner - start_inner))
        echo "Time taken for $f: $inner_duration seconds"
        sleep 5
    done

    end_outer=$(date +%s)
    outer_duration=$((end_outer - start_outer))
    echo "Total time taken for folder $d: $outer_duration seconds"
done
# sleep 60
# echo "finished sleeping"
