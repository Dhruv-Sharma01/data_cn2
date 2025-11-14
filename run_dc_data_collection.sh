#!/bin/bash

SCENARIOS=("a" "p" "c")
PROTOCOLS=("tcp" "udp")
SAMPLES_PER_PAIR=3
TEST_DURATION=30
STREAMS=(4 8 16)
TOPOLOGY_FILE="/opt/simulation/Sonic.json"
NODE_LIST_FILE="/opt/simulation/nodes_with_buffers.json"
HELPER_SCRIPT_DIR="/opt/simulation/scripts"
LOG_DIR_BASE="/opt/simulation/results"
echo "Starting Data Center Performance and Congestion Test..."
NODE_PAIRS=$(python3 ${HELPER_SCRIPT_DIR}/get_node_pairs.py ${NODE_LIST_FILE})
for protocol in "${PROTOCOLS[@]}"; do
    echo "[PROTOCOL: ${protocol}]"
    for scenario in "${SCENARIOS[@]}"; do
        echo "  [SCENARIO: ${scenario}]"
        echo "    Setting up Scenario ${scenario} environment..."
        ${HELPER_SCRIPT_DIR}/setup_scenario.sh ${scenario}
        sleep 15
        while IFS=, read -r c_name c_ip s_name s_ip; do
            for s_count in "${STREAMS[@]}"; do
                for ((i=1; i<=SAMPLES_PER_PAIR; i++)); do
                    echo "      TEST: ${c_name} -> ${s_name} (Sample $i, Streams $s_count)"
                    LOG_DIR="${LOG_DIR_BASE}/${protocol}_test/${scenario}"
                    mkdir -p ${LOG_DIR}
                    LOG_FILE="${LOG_DIR}/iperf_${protocol}_${c_name}_to_${s_name}_${scenario}_${s_count}.log"
                    ssh admin@${s_ip} "iperf3 -s -1 -D" > /dev/null
                    if [ "$protocol" == "tcp" ]; then
                        ssh admin@${c_ip} "iperf3 -c ${s_ip} -t ${TEST_DURATION} -P ${s_count} -J > ${LOG_FILE}"
                    else
                        ssh admin@${c_ip} "iperf3 -c ${s_ip} -u -b 0 -t ${TEST_DURATION} -P ${s_count} -J > ${LOG_FILE}"
                    fi
                    ssh admin@${s_ip} "killall iperf3" > /dev/null
                done
            done
        done <<< "$NODE_PAIRS"
        echo "    Cleaning up Scenario ${scenario}..."
        ${HELPER_SCRIPT_DIR}/setup_scenario.sh "cleanup"
    done
done
echo "All tests completed. Results are in ${LOG_DIR_BASE}"
