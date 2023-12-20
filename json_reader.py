from xml_reader_ts import xml_reader, champsim_config_reader, set_xml

import re
import pandas as pd
import json
import numpy as np

def parse_output(data, champsim_terminal_output):

    def grab_line_with_word(text, word):
        """
        This function grabs the line from a string that has a certain word in it.

        Args:
            text: The string to search.
            word: The word to search for.

        Returns:
            The line that contains the word, or None if the word is not found.
        """
        lines = text.splitlines()
        for line in lines:
            if word in line:
                return line
        return None
    results = []
    curr_instr = 0
    curr_cycles = 0
    instructions = []
    cycles = []
    cum_ipc = []
    for i in range(100):
        word = f"Simulation{i} complete"

        line = grab_line_with_word(champsim_terminal_output, word)
        line = line.replace(" ", "")
        result = re.search('IPC:(.*)', line)
        result = result.group(1)
        result = float(result.split("(")[0])
        results.append(result)

        instr = re.search('instructions:(.*)cycles', line)
        instr = int(instr.group(1))
        instructions.append(float(instr))

        cycle = re.search('cycles:(.*)cumulative', line)
        cycle = int(cycle.group(1))
        cycles.append(float(cycle))
        curr_instr += instr
        curr_cycles += cycle
        cum_ipc.append(curr_instr/curr_cycles)

    ######## TS #######
    total_res = []
    prev_res = {}
    champsim_info = champsim_config_reader("/Users/aaronchan/Documents/ChampSim/champsim_config.json")
    mpki = []
    def get_text_between(text, start, end):
        if end == "":
            pattern = rf"{re.escape(start)}(.*)"
        else:
            pattern = rf"{re.escape(start)}(.*?){re.escape(end)}"
        # Adjusting the pattern for your specific start and end strings
        
        match = re.search(pattern, text, re.DOTALL)

        # If a match is found, return the captured group
        if match:
            return match.group(1).strip()
        else:
            return "No text found between the specified strings"
    for i in range(len(data)):
        if i != 99:
            champsim_terminal_output_subset = get_text_between(champsim_terminal_output, f"=== Simulation{i} ===", f"=== Simulation{i+1} ===")
        else:
            champsim_terminal_output_subset = get_text_between(champsim_terminal_output, f"=== Simulation{i} ===", "")
        spec_data = data[i]
        res = xml_reader(spec_data, champsim_terminal_output_subset)
        #print(res["mpki"])
        mpki.append(res["mpki"])
        for key in res.keys():
            res[key] = float(res[key])
        if i != 0:
            for key in res.keys():
                res[key] += prev_res[key]
            #res += prev_res

        total_res.append(res)
        prev_res = res
    res_all = {}
    for res in total_res:
        for key in res.keys():
            value = res[key]
            if key not in res_all.keys():
                res_all[key] = [value]
            else:
                res_all[key].append(value)
    res_all["cumulative_ipc"] = cum_ipc
    res_all["mpki"] = mpki
    res_all["champsim_info"] = champsim_info
    return res_all


# with open("ts_2_out.txt", "r") as f:
#     output = f.read()
# with open("ts_2.json", "r") as f:
#     json_data = json.load(f)
# output_data = parse_output(json_data, output)
# key = list(output_data.keys())[0]

# def get_last_values(all_data):
#     last_values = {}
#     for key in all_data.keys():
#         if not isinstance(all_data[key], dict):
#             last_values[key] = str(int(all_data[key][-1]))
#         else:
#             for champsim_key in all_data[key].keys():
#                 last_values[champsim_key] = all_data[key][champsim_key]
#     return last_values
# my_data = get_last_values(output_data)
# print(my_data)
# print(key)
# set_xml(my_data, "/Users/aaronchan/Documents/ChampSim/input.xml", "output.xml")