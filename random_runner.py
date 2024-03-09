import numpy as np
import random
import json
import subprocess
import sys
import pickle
from datetime import datetime

# from json_reader import parse_output
import os
from google.cloud import firestore, storage
import time

frequency_mapper = {
    0: 1000,
    1: 1250,
    2: 1500,
    3: 1750,
    4: 2000,
    5: 2250,
    6: 2500,
    7: 2750,
    8: 3000,
    9: 3250,
    10: 3500,
    11: 3750,
    12: 4000,
    13: 4250,
    14: 4500,
}
ifetch_buffer_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
decode_buffer_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
dispatch_buffer_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
rob_size_mapper = {
    0: 16,
    1: 32,
    2: 48,
    3: 64,
    4: 80,
    5: 96,
    6: 112,
    7: 128,
    8: 144,
    9: 160,
    10: 176,
    11: 192,
    12: 208,
    13: 224,
    14: 240,
    15: 256,
    16: 320,
    17: 384,
    18: 448,
    19: 512,
    20: 576,
}
lq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
sq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
fetch_width_mapper = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
decode_width_mapper = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
dispatch_width_mapper = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
execute_width_mapper = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
lq_width_mapper = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
sq_width_mapper = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
retire_width_mapper = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
scheduler_size_mapper = {
    0: 2,
    1: 4,
    2: 8,
    3: 16,
    4: 32,
    5: 64,
    6: 128,
    7: 144,
    8: 160,
    9: 176,
    10: 192,
    11: 208,
    12: 256,
}
branch_predictor_mapper = {
    0: "bimodal",
    1: "gshare",
    2: "hashed_perceptron",
    3: "perceptron",
}
btb_mapper = {0: "basic_btb", 1: "basic_btb_1"}

# DIB
window_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64}
dib_sets_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64}
dib_ways_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64}

# L1I
l1i_sets_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64}
l1i_ways_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64}
l1i_rq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1i_wq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1i_pq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1i_mshr_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1i_prefetcher_mapper = {0: "no_instr", 1: "next_line_instr"}

# L1D
l1d_sets_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64}
l1d_ways_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64}
l1d_rq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1d_wq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1d_pq_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1d_mshr_size_mapper = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32, 5: 64, 6: 128}
l1d_prefetcher_mapper = {
    0: "ip_stride",
    1: "next_line",
    2: "no",
    3: "spp_dev",
    4: "va_ampm_lite",
}


def decode(act_encoded):
    act_decoded = {}
    if isinstance(act_encoded, dict):
        act_decoded["Frequency"] = frequency_mapper[act_encoded["Frequency"]]
        act_decoded["iFetchBufferSize"] = ifetch_buffer_size_mapper[
            act_encoded["iFetchBufferSize"]
        ]
        act_decoded["DecodeBufferSize"] = decode_buffer_size_mapper[
            act_encoded["DecodeBufferSize"]
        ]
        act_decoded["DispatchBufferSize"] = dispatch_buffer_size_mapper[
            act_encoded["DispatchBufferSize"]
        ]
        act_decoded["ROBSize"] = rob_size_mapper[act_encoded["ROBSize"]]
        act_decoded["LQSize"] = lq_size_mapper[act_encoded["LQSize"]]
        act_decoded["SQSize"] = sq_size_mapper[act_encoded["SQSize"]]
        act_decoded["FetchWidth"] = fetch_width_mapper[act_encoded["FetchWidth"]]
        act_decoded["DecodeWidth"] = decode_width_mapper[act_encoded["DecodeWidth"]]
        act_decoded["DispatchWidth"] = dispatch_width_mapper[
            act_encoded["DispatchWidth"]
        ]
        act_decoded["ExecuteWidth"] = execute_width_mapper[act_encoded["ExecuteWidth"]]
        act_decoded["LQWidth"] = lq_width_mapper[act_encoded["LQWidth"]]
        act_decoded["SQWidth"] = sq_width_mapper[act_encoded["SQWidth"]]
        act_decoded["RetireWidth"] = retire_width_mapper[act_encoded["RetireWidth"]]
        act_decoded["SchedulerSize"] = scheduler_size_mapper[
            act_encoded["SchedulerSize"]
        ]
        act_decoded["BranchPredictor"] = branch_predictor_mapper[
            act_encoded["BranchPredictor"]
        ]
        act_decoded["BTB"] = btb_mapper[act_encoded["BTB"]]

        act_decoded["DIBWindowSize"] = window_size_mapper[act_encoded["DIBWindowSize"]]
        act_decoded["DIBSets"] = dib_sets_mapper[act_encoded["DIBSets"]]
        act_decoded["DIBWays"] = dib_ways_mapper[act_encoded["DIBWays"]]

        act_decoded["L1ISets"] = l1i_sets_mapper[act_encoded["L1ISets"]]
        act_decoded["L1IWays"] = l1i_ways_mapper[act_encoded["L1IWays"]]
        act_decoded["L1IRQSize"] = l1i_rq_size_mapper[act_encoded["L1IRQSize"]]
        act_decoded["L1IWQSize"] = l1i_wq_size_mapper[act_encoded["L1IWQSize"]]
        act_decoded["L1IPQSize"] = l1i_pq_size_mapper[act_encoded["L1IPQSize"]]
        act_decoded["L1IMSHRSize"] = l1i_mshr_size_mapper[act_encoded["L1IMSHRSize"]]
        act_decoded["L1IPrefetcher"] = l1i_prefetcher_mapper[
            act_encoded["L1IPrefetcher"]
        ]

        act_decoded["L1DSets"] = l1i_sets_mapper[act_encoded["L1DSets"]]
        act_decoded["L1DWays"] = l1i_ways_mapper[act_encoded["L1DWays"]]
        act_decoded["L1DRQSize"] = l1i_rq_size_mapper[act_encoded["L1DRQSize"]]
        act_decoded["L1DWQSize"] = l1i_wq_size_mapper[act_encoded["L1DWQSize"]]
        act_decoded["L1DPQSize"] = l1i_pq_size_mapper[act_encoded["L1DPQSize"]]
        act_decoded["L1DMSHRSize"] = l1i_mshr_size_mapper[act_encoded["L1DMSHRSize"]]
        act_decoded["L1DPrefetcher"] = l1d_prefetcher_mapper[
            act_encoded["L1DPrefetcher"]
        ]
    return act_decoded


def select_config(trace):
    act_encoded = {}
    db = firestore.Client(project="nth-droplet-407821")

    doc_ref = db.collection("champsim").document(trace)
    doc = doc_ref.get()
    if not doc.exists:
        doc_ref.set({"dummy": "1"})
    # doc_ref.set({"8_1_2": "1"})
    found_good_value = False
    while not found_good_value:

        def pull_doc():
            data_store = db.collection("champsim").document(trace).get().to_dict()
            return data_store

        data_store = pull_doc()
        # repr(list(act_encoded.values())
        while (
            not act_encoded
            or ("_".join([str(i) for i in list(act_encoded.values())]) + ".txt")
            in data_store
        ):
            act_encoded["Frequency"] = random.randint(
                0, len(frequency_mapper.keys()) - 1
            )
            act_encoded["iFetchBufferSize"] = random.randint(
                0, len(ifetch_buffer_size_mapper.keys()) - 1
            )
            act_encoded["DecodeBufferSize"] = random.randint(
                0, len(decode_buffer_size_mapper.keys()) - 1
            )
            act_encoded["DispatchBufferSize"] = random.randint(
                0, len(dispatch_buffer_size_mapper.keys()) - 1
            )
            act_encoded["ROBSize"] = random.randint(0, len(rob_size_mapper.keys()) - 1)
            act_encoded["LQSize"] = random.randint(0, len(lq_size_mapper.keys()) - 1)
            act_encoded["SQSize"] = random.randint(0, len(sq_size_mapper.keys()) - 1)
            act_encoded["FetchWidth"] = random.randint(
                0, len(fetch_width_mapper.keys()) - 1
            )
            act_encoded["DecodeWidth"] = random.randint(
                0, len(decode_width_mapper.keys()) - 1
            )
            act_encoded["DispatchWidth"] = random.randint(
                0, len(dispatch_width_mapper.keys()) - 1
            )
            act_encoded["ExecuteWidth"] = random.randint(
                0, len(execute_width_mapper.keys()) - 1
            )
            act_encoded["LQWidth"] = random.randint(0, len(lq_width_mapper.keys()) - 1)
            act_encoded["SQWidth"] = random.randint(0, len(sq_width_mapper.keys()) - 1)
            act_encoded["RetireWidth"] = random.randint(
                0, len(retire_width_mapper.keys()) - 1
            )
            act_encoded["SchedulerSize"] = random.randint(
                0, len(scheduler_size_mapper.keys()) - 1
            )
            act_encoded["BranchPredictor"] = random.randint(
                0, len(branch_predictor_mapper.keys()) - 1
            )
            act_encoded["BTB"] = random.randint(0, len(btb_mapper.keys()) - 1)
            act_encoded["DIBWindowSize"] = random.randint(
                0, len(window_size_mapper.keys()) - 1
            )
            act_encoded["DIBSets"] = random.randint(0, len(dib_sets_mapper.keys()) - 1)
            act_encoded["DIBWays"] = random.randint(0, len(dib_ways_mapper.keys()) - 1)
            act_encoded["L1ISets"] = random.randint(0, len(l1i_sets_mapper.keys()) - 1)
            act_encoded["L1IWays"] = random.randint(0, len(l1i_ways_mapper.keys()) - 1)
            act_encoded["L1IRQSize"] = random.randint(
                0, len(l1i_rq_size_mapper.keys()) - 1
            )
            act_encoded["L1IWQSize"] = random.randint(
                0, len(l1i_wq_size_mapper.keys()) - 1
            )
            act_encoded["L1IPQSize"] = random.randint(
                0, len(l1i_pq_size_mapper.keys()) - 1
            )
            act_encoded["L1IMSHRSize"] = random.randint(
                0, len(l1i_mshr_size_mapper.keys()) - 1
            )
            act_encoded["L1IPrefetcher"] = random.randint(
                0, len(l1i_prefetcher_mapper.keys()) - 1
            )
            act_encoded["L1DSets"] = random.randint(0, len(l1d_sets_mapper.keys()) - 1)
            act_encoded["L1DWays"] = random.randint(0, len(l1d_ways_mapper.keys()) - 1)
            act_encoded["L1DRQSize"] = random.randint(
                0, len(l1d_rq_size_mapper.keys()) - 1
            )
            act_encoded["L1DWQSize"] = random.randint(
                0, len(l1d_wq_size_mapper.keys()) - 1
            )
            act_encoded["L1DPQSize"] = random.randint(
                0, len(l1d_pq_size_mapper.keys()) - 1
            )
            act_encoded["L1DMSHRSize"] = random.randint(
                0, len(l1d_mshr_size_mapper.keys()) - 1
            )
            act_encoded["L1DPrefetcher"] = random.randint(
                0, len(l1d_prefetcher_mapper.keys()) - 1
            )
        data_store_1 = pull_doc()
        # breakpoint()
        if repr(list(act_encoded.values())) not in data_store_1.keys():
            doc_ref = db.collection("champsim").document(trace)
            data_store_1[repr(list(act_encoded.values()))] = "1"
            doc_ref.set(data_store_1)
            found_good_value = True
            break
        break

    act_decoded = decode(act_encoded)
    write_to_json(act_decoded)
    return act_encoded


def write_to_json(action):
    champsim_ctrl_file = "champsim_config.json"
    with open(champsim_ctrl_file, "r") as JsonFile:
        data = json.load(JsonFile)
        data["ooo_cpu"][0]["frequency"] = action["Frequency"]
        data["ooo_cpu"][0]["ifetch_buffer_size"] = action["iFetchBufferSize"]
        data["ooo_cpu"][0]["decode_buffer_size"] = action["DecodeBufferSize"]
        data["ooo_cpu"][0]["dispatch_buffer_size"] = action["DispatchBufferSize"]
        data["ooo_cpu"][0]["rob_size"] = action["ROBSize"]
        data["ooo_cpu"][0]["lq_size"] = action["LQSize"]
        data["ooo_cpu"][0]["sq_size"] = action["SQSize"]
        data["ooo_cpu"][0]["fetch_width"] = action["FetchWidth"]
        data["ooo_cpu"][0]["decode_width"] = action["DecodeWidth"]
        data["ooo_cpu"][0]["dispatch_width"] = action["DispatchWidth"]
        data["ooo_cpu"][0]["execute_width"] = action["ExecuteWidth"]
        data["ooo_cpu"][0]["lq_width"] = action["LQWidth"]
        data["ooo_cpu"][0]["sq_width"] = action["SQWidth"]
        data["ooo_cpu"][0]["retire_width"] = action["RetireWidth"]
        data["ooo_cpu"][0]["scheduler_size"] = action["SchedulerSize"]
        data["ooo_cpu"][0]["branch_predictor"] = action["BranchPredictor"]
        data["ooo_cpu"][0]["btb"] = action["BTB"]

        data["DIB"]["window_size"] = action["DIBWindowSize"]
        data["DIB"]["sets"] = action["DIBSets"]
        data["DIB"]["ways"] = action["DIBWays"]

        data["L1I"]["sets"] = action["L1ISets"]
        data["L1I"]["ways"] = action["L1IWays"]
        data["L1I"]["rq_size"] = action["L1IRQSize"]
        data["L1I"]["wq_size"] = action["L1IWQSize"]
        data["L1I"]["pq_size"] = action["L1IPQSize"]
        data["L1I"]["mshr_size"] = action["L1IMSHRSize"]
        data["L1I"]["prefetcher"] = action["L1IPrefetcher"]

        data["L1D"]["sets"] = action["L1DSets"]
        data["L1D"]["ways"] = action["L1DWays"]
        data["L1D"]["rq_size"] = action["L1DRQSize"]
        data["L1D"]["wq_size"] = action["L1DWQSize"]
        data["L1D"]["pq_size"] = action["L1DPQSize"]
        data["L1D"]["mshr_size"] = action["L1DMSHRSize"]
        data["L1D"]["prefetcher"] = action["L1DPrefetcher"]
        with open(champsim_ctrl_file, "w") as JsonFile:
            json.dump(data, JsonFile, indent=4)


def run_program(iter, action_dict, trace):
    def current_datetime_to_numeric_representation():
        # Use a reference datetime (e.g., epoch)
        reference_datetime = datetime(2022, 1, 1)

        # Get the current datetime
        current_datetime = datetime.now()

        # Calculate total seconds elapsed since the reference datetime
        total_seconds = (current_datetime - reference_datetime).total_seconds()

        return total_seconds

    # Example usage:
    # numeric_representation = str(current_datetime_to_numeric_representation())
    name = list(action_dict.values())
    name = [str(n) for n in name]
    name = "_".join(name)
    trace_fp = f"traces/{trace}"
    process = subprocess.Popen(
        ["./config.sh", "champsim_config.json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate()
    if err.decode() == "":
        outstream = out.decode()
    else:
        print(err.decode())
        sys.exit()
    process = subprocess.Popen(
        ["make", "-j2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = process.communicate()
    if "error" not in err.decode():
        outstream = out.decode()
    else:
        print(err.decode())
        print(action_dict)
        sys.exit()
    print("Done configuring/making config")
    output_json_dir = f"output_json_{trace}"
    output_logs_dir = f"output_logs_{trace}"
    if not os.path.exists(output_json_dir):
        # Create the directory
        os.makedirs(output_json_dir)
    if not os.path.exists(output_logs_dir):
        # Create the directory
        os.makedirs(output_logs_dir)
    start_time = time.time()
    process = subprocess.Popen(
        [
            "./bin/champsim",
            "-w",
            "200000000",
            "--simulation-instructions",
            "500000000",
            trace_fp,
            "--json",
            f"{output_json_dir}/{name}.json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = process.communicate()
    print("Done running program, time to run:", (time.time() - start_time) / 60)
    if err.decode() == "":
        outstream = out.decode()
    else:
        print(err.decode())
        print(print(action_dict))
        # sys.exit()
    if len(outstream) < 100:
        print(outstream)
    txt_file_path = f"{output_logs_dir}/{name}.txt"
    json_file_path = f"{output_json_dir}/{name}.json"
    with open(f"{output_logs_dir}/{name}.txt", "w+") as file:
        file.write(outstream)
    with open(f"{output_json_dir}/{name}.json", "r") as json_file:
        json_data = json.load(json_file)

    client = storage.Client(
        project="nth-droplet-407821"
    )  # storage.Client.from_service_account_json('nth-droplet-407821-515306f939e3.json')

    # Specify the bucket name and document path
    bucket_name = "champsim"

    # Get a reference to the bucket
    bucket = client.bucket(bucket_name)

    # Upload the document from the local file
    json_blob = bucket.blob(json_file_path)
    json_blob.upload_from_filename(json_file_path)

    txt_blob = bucket.blob(txt_file_path)
    txt_blob.upload_from_filename(txt_file_path)
    # data_store[name] = parse_output(json_data, outstream)
    print("Done storing everything")


# try:
#     with open('champsim_data.pickle', 'rb') as handle:
#         data_store = pickle.load(handle)
# except:
#     data_store = {}

"""
import subprocess
import os

def download_with_subprocess(url, destination_file=None):
    if destination_file is None:
        destination_file = os.path.basename(url)  # Use the filename from the URL

    wget_command = ['wget', url, '-O', destination_file]
    subprocess.run(wget_command) 

# Example
url = "https://www.example.com/images/sample_image.jpg"
download_with_subprocess(url) 

"""
def download_with_subprocess(url, destination_file=None):
    if destination_file is None:
        destination_file = os.path.basename(url)  # Use the filename from the URL

    wget_command = ['wget', url, '-O', destination_file]
    subprocess.run(wget_command) 

def main(iter, trace):
    action_dict = select_config(trace)
    run_program(iter, action_dict, trace)


traces = [
    "400.perlbench-41B.champsimtrace.xz",
    "400.perlbench-50B.champsimtrace.xz",
    "401.bzip2-226B.champsimtrace.xz",
    "401.bzip2-277B.champsimtrace.xz",
    "401.bzip2-38B.champsimtrace.xz",
    "401.bzip2-7B.champsimtrace.xz",
    "403.gcc-16B.champsimtrace.xz",
    "403.gcc-17B.champsimtrace.xz",
    "403.gcc-48B.champsimtrace.xz",
    "410.bwaves-1963B.champsimtrace.xz",
    "410.bwaves-2097B.champsimtrace.xz",
    "410.bwaves-945B.champsimtrace.xz",
    "416.gamess-875B.champsimtrace.xz",
    "429.mcf-184B.champsimtrace.xz",
    "429.mcf-192B.champsimtrace.xz",
    "429.mcf-217B.champsimtrace.xz",
    "429.mcf-22B.champsimtrace.xz",
    "429.mcf-51B.champsimtrace.xz",
    "433.milc-127B.champsimtrace.xz",
    "433.milc-274B.champsimtrace.xz",
    "433.milc-337B.champsimtrace.xz",
    "434.zeusmp-10B.champsimtrace.xz",
    "435.gromacs-111B.champsimtrace.xz",
    "435.gromacs-134B.champsimtrace.xz",
    "435.gromacs-226B.champsimtrace.xz",
    "435.gromacs-228B.champsimtrace.xz",
    "436.cactusADM-1804B.champsimtrace.xz",
    "437.leslie3d-134B.champsimtrace.xz",
    "437.leslie3d-149B.champsimtrace.xz",
    "437.leslie3d-232B.champsimtrace.xz",
    "437.leslie3d-265B.champsimtrace.xz",
    "437.leslie3d-271B.champsimtrace.xz",
    "437.leslie3d-273B.champsimtrace.xz",
    "444.namd-120B.champsimtrace.xz",
    "444.namd-166B.champsimtrace.xz",
    "444.namd-23B.champsimtrace.xz",
    "444.namd-321B.champsimtrace.xz",
    "444.namd-33B.champsimtrace.xz",
    "444.namd-426B.champsimtrace.xz",
    "444.namd-44B.champsimtrace.xz",
    "445.gobmk-17B.champsimtrace.xz",
    "445.gobmk-2B.champsimtrace.xz",
    "445.gobmk-30B.champsimtrace.xz",
    "445.gobmk-36B.champsimtrace.xz",
    "447.dealII-3B.champsimtrace.xz",
    "450.soplex-247B.champsimtrace.xz",
    "450.soplex-92B.champsimtrace.xz",
    "453.povray-252B.champsimtrace.xz",
    "453.povray-576B.champsimtrace.xz",
    "453.povray-800B.champsimtrace.xz",
    "453.povray-887B.champsimtrace.xz",
    "454.calculix-104B.champsimtrace.xz",
    "454.calculix-460B.champsimtrace.xz",
    "456.hmmer-191B.champsimtrace.xz",
    "456.hmmer-327B.champsimtrace.xz",
    "456.hmmer-88B.champsimtrace.xz",
    "458.sjeng-1088B.champsimtrace.xz",
    "458.sjeng-283B.champsimtrace.xz",
    "458.sjeng-31B.champsimtrace.xz",
    "458.sjeng-767B.champsimtrace.xz",
    "459.GemsFDTD-1169B.champsimtrace.xz",
    "459.GemsFDTD-1211B.champsimtrace.xz",
    "459.GemsFDTD-1320B.champsimtrace.xz",
    "459.GemsFDTD-1418B.champsimtrace.xz",
    "459.GemsFDTD-1491B.champsimtrace.xz",
    "459.GemsFDTD-765B.champsimtrace.xz",
    "462.libquantum-1343B.champsimtrace.xz",
    "462.libquantum-714B.champsimtrace.xz",
    "464.h264ref-30B.champsimtrace.xz",
    "464.h264ref-57B.champsimtrace.xz",
    "464.h264ref-64B.champsimtrace.xz",
    "464.h264ref-97B.champsimtrace.xz",
    "465.tonto-1914B.champsimtrace.xz",
    "465.tonto-44B.champsimtrace.xz",
    "470.lbm-1274B.champsimtrace.xz",
    "471.omnetpp-188B.champsimtrace.xz",
    "473.astar-153B.champsimtrace.xz",
    "473.astar-359B.champsimtrace.xz",
    "473.astar-42B.champsimtrace.xz",
    "481.wrf-1170B.champsimtrace.xz",
    "481.wrf-1254B.champsimtrace.xz",
    "481.wrf-1281B.champsimtrace.xz",
    "481.wrf-196B.champsimtrace.xz",
    "481.wrf-455B.champsimtrace.xz",
    "481.wrf-816B.champsimtrace.xz",
    "482.sphinx3-1100B.champsimtrace.xz",
    "482.sphinx3-1297B.champsimtrace.xz",
    "482.sphinx3-1395B.champsimtrace.xz",
    "482.sphinx3-1522B.champsimtrace.xz",
    "482.sphinx3-234B.champsimtrace.xz",
    "482.sphinx3-417B.champsimtrace.xz",
    "483.xalancbmk-127B.champsimtrace.xz",
    "483.xalancbmk-716B.champsimtrace.xz",
    "483.xalancbmk-736B.champsimtrace.xz",
]
for trace in traces:
    url = f"https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/{trace}"
    trace_fp = f"traces/{trace}"
    download_with_subprocess(url, trace_fp)
    for i in range(5):
        print(i)
        main(i, trace)
        try:
            main(i, trace)
        except KeyboardInterrupt:
            # quit
            sys.exit()
        except Exception as ex:
            print("ERROR:", ex)
            continue
    os.remove(trace_fp) # clean up the folder so we don't have all the traces

# with open("champsim_data.pickle", "wb+") as handle:
#     pickle.dump(data_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
