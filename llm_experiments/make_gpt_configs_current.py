# make gpt configs
'''
Let's change the gpt config script to use some of the ideas from the timeseries config script.
These scripts make json configs for experiments.
The timeseries file has undergone some changes to allow it to have greater lookahead. I want to do the same for the gpt one.
We will need to fix it at one specific rate (let's take the 20_000 budget point) instead of the larger list we have now.
Once that is done we will change to allow larger lookahead.
Let's start with all powers of 10 up to 10e6 and also them multiplied by 2.5 and 5.
Once we have those larger lookahead windows, we will need to extend the deadline time to actuall allow that much lookahead.
We need to scale it accordingly.
We also need to scale the number of max items accordingly.
'''
import json
import os

TOTAL_OBJS = 3270;
SKIP_RATIO = 0.5;
SKIP_AMOUNT = int(TOTAL_OBJS * SKIP_RATIO);
REMAINING_AMOUNT = TOTAL_OBJS - SKIP_AMOUNT;

# base_logfile_dir = "log_outputs4"
base_logfile_dir = "log_outputs"

# strategies = ["aquifer_greedy", "aquifer_optimal", "big", "small", "tiny", "eddies"]
strategies = ["aquifer_greedy", "aquifer_optimal"]

preclassifiers = [
    ('agreement0','./llm_preclassifiers/llm_agreement_0.json'),
    ('agreement1','./llm_preclassifiers/llm_agreement_1.json'),
    ('overtake0','./llm_preclassifiers/llm_overtake_0.json'),
    ('overtake1','./llm_preclassifiers/llm_overtake_1.json'),
    ('n_way','./llm_preclassifiers/llm_individual_correctness.json'),
    ('min_necessary','./llm_preclassifiers/llm_minimum_necessary.json'),
]

# we fix the time rate
# but we will vary the money budget
# input_delay_micros = 25000
input_delay_micros = 25000 * 2
target_time_micros = input_delay_micros
delay_time_micros = input_delay_micros
items_per_second = 1_000_000 / target_time_micros
items_per_dollar = 20_000

# look ahead 1 dollar
dollar_deadline = 1.0

def compute_budget_vars(input_delay_micros, dollar_deadline, starting_items_per_dollar, desired_lookahead):
    # use inptu delay, dollar deadline, and starting_items_per_dollar to get the time for the whole dollar
    seconds_per_item = input_delay_micros / 1_000_000
    items_per_usd = starting_items_per_dollar
    seconds_per_usd = items_per_usd * seconds_per_item    

    # we could normally compute this many items but we need *this* many items
    scale_ratio = desired_lookahead / starting_items_per_dollar

    money_budget_per_deadline = dollar_deadline * scale_ratio
    deadline_window_ms = int(seconds_per_usd * 1000 * scale_ratio)
    # print(f"Seconds per item: {seconds_per_item}")
    # print(f"Items per dollar: {items_per_usd}")
    # print(f"Seconds per dollar: {seconds_per_usd}")
    # print(f"Scale ratio: {scale_ratio}")
    # print(f"Money budget per deadline: {money_budget_per_deadline}")
    # print(f"Deadline window ms: {deadline_window_ms}")
    # print(f"Items per second: {items_per_second}")
    # print(f"Items per deadline: {items_per_deadline}")
    # print(f"Delay time micros: {delay_time_micros}")
    # print(f"Input delay micros: {input_delay_micros}")
    # print(f"Target time micros: {target_time_micros}")
    # print(f"Budgets per item: {budgets_per_item}")
    return money_budget_per_deadline, deadline_window_ms


# deadline_window_ms = 1_000

# list_items_per_dollar = [
#     20_000,
#     # 25_000,
#     # 30_000,
# ]

# budgets_per_item = [1/v for v in list_items_per_dollar]

budgets_per_item = [1/v for v in [
    500,
    750,
    1_000,
    1_500,
    2_000,
    5_000,
    10_000,
    15_000,
    20_000,
    25_000,
    30_000,
    40_000,
    45_000,
    50_000,
    55_000,
    60_000,
    65_000,
    70_000,
    75_000,
    80_000,
    85_000,
    90_000,
    95_000,
    100_000,
    125_000,
    150_000,
    175_000,
    200_000,
]]

budgets_per_item = [1/v for v in [ 20_000 ] ]

# greedy_lookahead_window_sizes = []
# # for i in range(0, 8):
# # stop at 10_000
# import numpy as np
# # for i in range(0, int(np.log10(10_000))+1):
# for i in range(0, int(np.log10(10_000))+0):
#     greedy_lookahead_window_sizes.append(10 ** i)
#     greedy_lookahead_window_sizes.append((10 ** i) * 2.5)
#     greedy_lookahead_window_sizes.append((10 ** i) * 5)
#     greedy_lookahead_window_sizes.append((10 ** i) * 7.5)

# greedy_lookahead_window_sizes = sorted(list(int(v) for v in set(greedy_lookahead_window_sizes) if v <= 10**6))
# greedy_lookahead_window_sizes = [10, 100, 1_000, 10_000]
greedy_lookahead_window_sizes = [5, 10, 20, 30, 40, 50, 100, 500, 1000]
print(f"Greedy lookahead window sizes ({len(greedy_lookahead_window_sizes)}): {greedy_lookahead_window_sizes}")
# exit()
new_folder_name = "llm_main_experiment_combined_configs"
os.makedirs(new_folder_name, exist_ok=True)


# rust side
# struct GptExperimentConfig {
#     "boolq_file": String,
#     "gpt_results_path": String,
#     "cached_embedding_path": String,
#     "query_path": String,
#     "max_total_samples": Option<usize>,
#     "history_window_size": Option<usize>,
#     "greedy_lookahead_window_size": Option<usize>,
#     "optimal_lookahead_window_size": Option<usize>,
#     "deadline_window_ms": Option<u64>,
#     "money_budget_per_deadline": f64,
#     "target_time_micros": Option<Delay>,
#     "input_delay_micros": Option<Delay>,
#     "overall_time_limit_ms": Option<u64>,
#     "initial_startup_delay_ms": Option<u64>,
#     "routing_strategy": Option<RoutingOptions>,
#     "log_folder": Option<HabString>,
# }

baseline_config = {
    "log_folder": "log_outputs/example",
    "bucket_path": ",/llm_preclassifiers/llm_agreement_0.json",
    "boolq_file": "boolq_dataset_dev.jsonl",
    "gpt_results_path": "./gpt_results.json",
    "cached_embedding_path": "question_embeddings.npy",
    "max_total_samples": REMAINING_AMOUNT,
    "query_path": "./test_queries_deployment/gpt_query.json",
    "initial_startup_delay_ms": 5000,
    "money_budget_per_deadline": 1.0,
    "deadline_window_ms": 1_000,
    "target_time_micros": delay_time_micros,
    "input_delay_micros": delay_time_micros,
    "greedy_lookahead_window_size": 5,
    "optimal_lookahead_window_size": 5,
    "overall_time_limit_ms": 500000,
    "routing_strategy": "aquifer_greedy"
}


# strategy = "aquifer_greedy"
# budget_index = 0
for budget_index in range(len(budgets_per_item)):
    for (pclass_name, pclass_path) in preclassifiers:
        baseline_config["bucket_path"] = pclass_path
        for window_size in greedy_lookahead_window_sizes[:]:
            sample_budget_per_item = budgets_per_item[budget_index]

            budget_per_deadline, deadline_window_ms = compute_budget_vars(input_delay_micros, dollar_deadline, items_per_dollar, window_size)
            print(f"Window size: {window_size}")
            print(f"Budget per item: {budgets_per_item[budget_index]}")
            print(f"Budget per deadline: {budget_per_deadline}")
            print(f"Deadline window ms: {deadline_window_ms}")

            for strategy in strategies[:]:
                if "opt" in strategy and window_size > 5:
                    continue

                config = baseline_config.copy()
                config["routing_strategy"] = strategy
                config["money_budget_per_deadline"] = budget_per_deadline
                config["deadline_window_ms"] = deadline_window_ms
                config["greedy_lookahead_window_size"] = window_size
                config["optimal_lookahead_window_size"] = window_size

                logging_folder = f"{base_logfile_dir}/{strategy}__{pclass_name}__window_size_{window_size}__budget_{budget_index}"
                # os.makedirs(os.path.join("gpt_query", logging_folder), exist_ok=True)
                os.makedirs(logging_folder, exist_ok=True)
                config["log_folder"] = f"{logging_folder}"

                config_name = f"{strategy}__{pclass_name}__window_size_{window_size}__budget_{budget_index}.json"
                with open(os.path.join(new_folder_name, config_name), 'w') as f:
                # with None as f:
                # if True:
                    json.dump(config, f, indent=4)
                    print(f"Configuration saved to {base_logfile_dir}/{config_name}")

            print()

