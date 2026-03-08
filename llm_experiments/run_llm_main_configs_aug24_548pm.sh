#!/bin/bash
python make_gpt_configs_current.py
for config_file_path in llm_main_experiment_combined_configs/*overtake0*; do
    echo "now using $config_file_path"
    time ./gpt_query "$config_file_path" log_config.yml
    mv log/output.log log_outputs/$(basename $config_file_path).log
done
