#!/usr/bin/env bash
#
# setup.sh
#
# Combined helper + mini-orchestrator for Scenario setup/cleanup and running one round.
#
# Usage:
#   setup.sh A|B|C|cleanup
#     -> behaves as the helper script your driver expects (setup or cleanup a single scenario)
#
#   setup.sh one_round
#     -> runs the data-collection driver once with SAMPLES_PER_PAIR=1 (so one run per pair),
#        letting the driver call this script for each scenario. The driver path can be set by
#        RUN_DRIVER (default: ../run_dc_data_collection.sh or /opt/simulation/run_dc_data_collection.sh).
#
# Environment variables (all optional, sensible defaults provided):
#   NODE_LIST_FILE  - path to nodes_with_buffers.json (default /opt/simulation/nodes_with_buffers.json)
#   SSH_USER        - user to ssh into nodes (default admin)
#   BG_STREAMS      - number of background streams for Scenario B (default 16)
#   BG_DURATION     - duration for background flows in seconds (default 600)
#   DEFAULT_DEST_LEAF - name of destination leaf to stress (default leaf02)
#   RUN_DRIVER      - path to run_dc_data_collection.sh for one_round mode (default /opt/simulation/run_dc_data_collection.sh)
#
set -euo pipefail
IFS=$'\n\t'

NODE_LIST_FILE="${NODE_LIST_FILE:-/opt/simulation/nodes_with_buffers.json}"
SSH_USER="${SSH_USER:-admin}"
STATE_FILE="/tmp/setup_scenario_state.json"
BG_STREAMS="${BG_STREAMS:-16}"
BG_DURATION="${BG_DURATION:-600}"
DEFAULT_DEST_LEAF="${DEFAULT_DEST_LEAF:-leaf02}"
RUN_DRIVER="${RUN_DRIVER:-/opt/simulation/run_dc_data_collection.sh}"

function log() { echo ">> $(date +'%F %T') - $*"; }

# ------- JSON helpers (jq required) -------
function get_leaf_nodes() {
  jq -r '.nodes[] | select(.name|test("^leaf")) | "\(.name) \(.ip)"' "$NODE_LIST_FILE"
}
function get_spine_nodes() {
  jq -r '.nodes[] | select(.name|test("^spine")) | "\(.name) \(.ip)"' "$NODE_LIST_FILE"
}
function get_node_ip_by_name() {
  local name="$1"
  jq -r --arg nm "$name" '.nodes[] | select(.name==$nm) | .ip' "$NODE_LIST_FILE"
}

# ------- state helpers -------
function read_state() {
  if [[ -f "$STATE_FILE" ]]; then
    cat "$STATE_FILE"
  else
    echo "{}"
  fi
}
function write_state() {
  local content="$1"
  echo "$content" > "$STATE_FILE"
}

function cleanup_all() {
  log "Running cleanup..."
  if [[ -f "$STATE_FILE" ]]; then
    # kill remote pids if any
    jq -c '.bg_processes[]?' "$STATE_FILE" 2>/dev/null | while read -r entry; do
      node=$(echo "$entry" | jq -r '.node')
      ip=$(echo "$entry" | jq -r '.ip')
      pid=$(echo "$entry" | jq -r '.pid')
      if [[ -n "$pid" && "$pid" != "null" ]]; then
        log "Killing remote bg pid $pid on $node($ip)"
        ssh "${SSH_USER}@${ip}" "kill -9 ${pid} >/dev/null 2>&1 || true" || true
      fi
    done || true

    # remove iptables rules (for failures)
    jq -c '.iptables_rules[]?' "$STATE_FILE" 2>/dev/null | while read -r entry; do
      node=$(echo "$entry" | jq -r '.node')
      ip=$(echo "$entry" | jq -r '.ip')
      dst=$(echo "$entry" | jq -r '.dst')
      if [[ -n "$dst" && "$dst" != "null" ]]; then
        log "Removing iptables DROP for dst $dst on $node($ip)"
        ssh "${SSH_USER}@${ip}" "iptables -D FORWARD -d ${dst} -j DROP 2>/dev/null || true" || true
      fi
    done || true

    rm -f "$STATE_FILE"
  else
    log "No state file found, nothing to cleanup."
  fi
  log "Cleanup done."
}

# ------- scenario setups -------
function setup_A() {
  log "Setting up Scenario A (baseline)."
  cleanup_all
  # ensure no iperf3 processes remain on leaf nodes
  get_leaf_nodes | while read -r ln; do
    node=$(echo "$ln" | awk '{print $1}')
    ip=$(echo "$ln" | awk '{print $2}')
    log "Ensuring no iperf3 on $node($ip)"
    ssh "${SSH_USER}@${ip}" "pkill -f iperf3 || true" || true
  done
  log "Scenario A ready (baseline)."
}

function setup_B() {
  log "Setting up Scenario B (congested/hotspot)."

  cleanup_all

  # pick destination leaf ip
  local dest_leaf_ip
  if get_node_ip_by_name "$DEFAULT_DEST_LEAF" >/dev/null 2>&1; then
    dest_leaf_ip="$(get_node_ip_by_name "$DEFAULT_DEST_LEAF")"
  else
    dest_leaf_ip="$(get_leaf_nodes | head -n1 | awk '{print $2}')"
  fi
  log "Chosen congestion target = $dest_leaf_ip"

  # start background iperf flows from all leafs except dest
  local bg_json="[]"
  get_leaf_nodes | while read -r ln; do
    node=$(echo "$ln" | awk '{print $1}')
    ip=$(echo "$ln" | awk '{print $2}')
    if [[ "$ip" == "$dest_leaf_ip" ]]; then
      continue
    fi

    log "Starting background iperf from $node($ip) -> $dest_leaf_ip with ${BG_STREAMS} streams (duration ${BG_DURATION}s)"
    remote_cmd="nohup iperf3 -c ${dest_leaf_ip} -t ${BG_DURATION} -P ${BG_STREAMS} --json > /tmp/bg_${node}_to_target.json 2>&1 & echo \$!"
    pid=$(ssh "${SSH_USER}@${ip}" "$remote_cmd" 2>/dev/null || true)
    pid="${pid//[$'\t\r\n ']}"  # trim
    if [[ -z "$pid" ]]; then pid=null; fi

    entry=$(jq -n --arg node "$node" --arg ip "$ip" --arg pid "$pid" '{node:$node, ip:$ip, pid:$pid}')
    bg_json=$(echo "$bg_json" | jq ". + [ $entry ]")
  done

  state=$(jq -n --arg dest "$dest_leaf_ip" '{scenario:"B", bg_processes: [], iptables_rules: [] }')
  state=$(echo "$state" | jq --argjson bg "$bg_json" '.bg_processes = $bg')
  write_state "$state"

  log "Background flows started for Scenario B."
}

function setup_C() {
  log "Setting up Scenario C (fault tolerance)."

  cleanup_all

  # pick destination
  local dest_leaf_ip
  if get_node_ip_by_name "$DEFAULT_DEST_LEAF" >/dev/null 2>&1; then
    dest_leaf_ip="$(get_node_ip_by_name "$DEFAULT_DEST_LEAF")"
  else
    dest_leaf_ip="$(get_leaf_nodes | head -n1 | awk '{print $2}')"
  fi
  log "Destination for failure simulation = $dest_leaf_ip"

  # pick 1-2 spine nodes (random if shuf available)
  local chosen_lines
  if command -v shuf >/dev/null 2>&1; then
    mapfile -t chosen_lines < <(get_spine_nodes | shuf -n 2)
  else
    mapfile -t chosen_lines < <(get_spine_nodes | head -n 2)
  fi

  iptables_json="[]"
  for ln in "${chosen_lines[@]}"; do
    node=$(echo "$ln" | awk '{print $1}')
    ip=$(echo "$ln" | awk '{print $2}')
    log "Inserting DROP rule on spine $node($ip) for dst $dest_leaf_ip (simulate link down)"
    ssh "${SSH_USER}@${ip}" "iptables -A FORWARD -d ${dest_leaf_ip} -j DROP" || true
    entry=$(jq -n --arg node "$node" --arg ip "$ip" --arg dst "$dest_leaf_ip" '{node:$node, ip:$ip, dst:$dst}')
    iptables_json=$(echo "$iptables_json" | jq ". + [ $entry ]")
  done

  state=$(jq -n --arg scenario "C" '{scenario:$scenario, bg_processes: [], iptables_rules: [] }')
  state=$(echo "$state" | jq --argjson ipt "$iptables_json" '.iptables_rules = $ipt')
  write_state "$state"

  log "Scenario C setup complete (iptables DROP rules deployed)."
}

# orchestrator: run one round by calling the driver with SAMPLES_PER_PAIR=1
function run_one_round_driver() {
  if [[ ! -x "$RUN_DRIVER" ]]; then
    log "Driver script not found or not executable: $RUN_DRIVER"
    log "Please set RUN_DRIVER to point to your run_dc_data_collection.sh"
    return 1
  fi

  # export minimal env to force one sample per pair
  export SAMPLES_PER_PAIR=1
  export TEST_DURATION="${TEST_DURATION:-30}"
  export STREAMS="${STREAMS:-4}"

  log "Invoking driver $RUN_DRIVER with SAMPLES_PER_PAIR=1 ..."
  bash "$RUN_DRIVER"
  log "Driver run finished."
}

# main dispatcher
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 A|B|C|cleanup|one_round"
  exit 2
fi

cmd="$1"
case "$cmd" in
  A|a)
    setup_A
    ;;
  B|b)
    setup_B
    ;;
  C|c)
    setup_C
    ;;
  cleanup)
    cleanup_all
    ;;
  one_round)
    # run one round: simply call the driver; the driver will call this script for setups per scenario
    run_one_round_driver
    ;;
  *)
    echo "Unknown command: $cmd"
    echo "Usage: $0 A|B|C|cleanup|one_round"
    exit 2
    ;;
esac

exit 0
