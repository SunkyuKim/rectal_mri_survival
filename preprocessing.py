import pandas as pd
import json
import sys, os
from datetime import date
from dataloader import DataLoader

def str_to_date(datestr):
    return date(int(datestr.split("-")[0]), int(datestr.split("-")[1]), int(datestr.split("-")[2]))

def generate_overall_survival_labels(patient_samples_dict, save_path=None):
    overall_survival_labels = {}
    for research_no in patient_samples_dict.keys():
        patient_samples = patient_samples_dict[research_no]
        first_date = "9999-99-99"
        last_date = "0000-00-00"
        censored = True
        for sample in patient_samples:
            if sample['first_date'] < first_date:
                first_date = sample['first_date']
            if sample['last_date'] > last_date:
                last_date = sample['last_date']
            if sample['survival'] == "False":
                censored = False
        f_date = str_to_date(first_date)
        l_date = str_to_date(last_date)
        date_difference = (l_date - f_date).days
        overall_sample = {
            'research_no': research_no,
            'first_date': first_date,
            'last_date': last_date,
            'survival_days': str(date_difference),
            'censored': str(censored)
        }
        overall_survival_labels[research_no] = overall_sample

    if save_path is not None:
        with open(save_path, 'w') as wf:
            json.dump(overall_survival_labels, wf)

    print("GENERATED:", len(overall_survival_labels.keys()), "samples - overall survival labels-", save_path)

    return overall_survival_labels

def generate_overall_survival_data(
        overall_survival_labels,
        mri_samples_by_research_no,
        patients_records=None,
        cea_records=None,
        op_code_records=None,
        op_code_binaries=None,
        rt_records=None,
        save_path=None):

    overall_survival_data = []


    valid_keys = set(ovrall_survival_labels.keys()).intersection(set(mri_samples_by_research_no.keys()))


    unique_op_code_list, id2opcode, opcode2id = op_code_binaries


    rt_size, id2rtname, rtname2id = rt_statistics(rt_records)

    for research_no in valid_keys:
        overall_survival_label = overall_survival_labels[research_no]
        if research_no in mri_samples_by_research_no.keys():
            mri_samples = mri_samples_by_research_no[research_no]
            for mri_sample in mri_samples:
                target_date = mri_sample['reading_date']

                # CEA
                if research_no in cea_records.keys():
                    cea_items = cea_records[research_no]
                    closest_cea_idx = find_closest_cea(target_date, cea_items)
                    cea_value = cea_records[research_no][closest_cea_idx]['cea_unit_value']
                else:
                    cea_value = 0

                # OP
                target_op_codes = ['0'] * len(unique_op_code_list)
                if research_no in op_code_records.keys():
                    for item in op_code_records[research_no]:
                        if item['op_date'] <= target_date:
                            target_op_codes[opcode2id[item['op_code']]] = '1'

                # RT
                rt_array = ['0'] * (1 + rt_size)
                if research_no in rt_records.keys():
                    rt_items = rt_records[research_no]
                    for rt_item in rt_items:
                        rt_date = rt_item['rt_date']
                        if rt_date < target_date:
                            rt_array[0] = '1'
                            rt_name = rt_item['rt_name']
                            rt_array[1 + rtname2id[rt_name]] = '1'

                # Survival
                last_date = overall_survival_label['last_date']
                l_date = str_to_date(last_date)
                t_date = str_to_date(target_date)
                date_difference = (l_date - t_date).days
                if date_difference < 0:
                    continue


                overall_survival_sample = {
                    'research_no': research_no,
                    'gender': [str(int(patients_records[research_no][0]['gender'] == 'F')), str(int(patients_records[research_no][0]['gender'] == 'M'))],
                    'age': str(int(patients_records[research_no][0]['first_age'])/100),
                    'CEA': str(cea_value),
                    'OP': target_op_codes,
                    'RT': rt_array,
                    'reading_text': mri_sample['reading_text'],
                    'reading_date': target_date,
                    'last_date': last_date,
                    'survival_days': str(date_difference),
                    'censored': overall_survival_label['censored']
                }
                overall_survival_data.append(overall_survival_sample)


    if save_path is not None:
        with open(save_path, 'w') as wf:
            json.dump(overall_survival_data, wf)

    print("GENERATED:", len(overall_survival_data), "samples - for training")
    return overall_survival_data



def generate_overall_survival_data_with_mri_cea_op_rt(overall_survival_labels, mri_samples_by_research_no, patients_records, cea_records, op_code_records, op_code_binaries, rt_records, save_path=None):
    overall_survival_data = []

    unique_op_code_list, id2opcode, opcode2id = op_code_binaries

    keys_dict = dict()
    keys_dict['os'] = set(overall_survival_labels.keys())
    keys_dict['mri_keys'] = set(mri_samples_by_research_no.keys())
    keys_dict['patients_keys'] = set(patients_records.keys())
    keys_dict['cea_keys'] = set(cea_records.keys())

    valid_keys = set.intersection(*keys_dict.values())

    rt_size, id2rtname, rtname2id = rt_statistics(rt_records)

    for research_no in valid_keys:
        overall_survival_label = overall_survival_labels[research_no]
        if research_no in mri_samples_by_research_no.keys():
            mri_samples = mri_samples_by_research_no[research_no]
            for mri_sample in mri_samples:
                target_date = mri_sample['reading_date']
                if research_no in cea_records.keys():
                    cea_items = cea_records[research_no]
                    closest_cea_idx = find_closest_cea(target_date, cea_items)
                    cea_value = cea_records[research_no][closest_cea_idx]['cea_unit_value']
                else:
                    cea_value = 0

                target_op_codes = ['0'] * len(unique_op_code_list)
                if research_no in op_code_records.keys():
                    for item in op_code_records[research_no]:
                        if item['op_date'] <= target_date:
                            target_op_codes[opcode2id[item['op_code']]] = '1'

                rt_array = ['0'] * (1 + rt_size)
                if research_no in rt_records.keys():
                    rt_items = rt_records[research_no]
                    for rt_item in rt_items:
                        rt_date = rt_item['rt_date']
                        if rt_date < target_date:
                            rt_array[0] = '1'
                            rt_name = rt_item['rt_name']
                            rt_array[1 + rtname2id[rt_name]] = '1'

                last_date = overall_survival_label['last_date']
                l_date = str_to_date(last_date)
                t_date = str_to_date(target_date)
                date_difference = (l_date - t_date).days

                if date_difference < 0:
                    continue

                overall_survival_sample = {
                    'research_no': research_no,
                    'gender': [str(int(patients_records[research_no][0]['gender'] == 'F')), str(int(patients_records[research_no][0]['gender'] == 'M'))],
                    'age': str(int(patients_records[research_no][0]['first_age'])/100),
                    'CEA': str(cea_value),
                    'OP': target_op_codes,
                    'RT': rt_array,
                    'reading_text': mri_sample['reading_text'],
                    'reading_date': target_date,
                    'last_date': last_date,
                    # 'survival_days': overall_survival_label['survival_days'],
                    'survival_days': str(date_difference),
                    'censored': overall_survival_label['censored']
                }

                overall_survival_data.append(overall_survival_sample)
    if save_path is not None:
        with open(save_path, 'w') as wf:
            json.dump(overall_survival_data, wf)


    print("GENERATED:", len(overall_survival_data), "samples - for training")
    return overall_survival_data



def unique_list_in_column(dict, column_name):
    res = []
    for key in dict.keys():
        for item in dict[key]:
            c_value = item[column_name]
            if c_value not in res:
                res.append(c_value)

    id2item = {idx: res[idx] for idx in range(len(res))}
    item2id = {v: k for k, v in id2item.items()}
    print("Unique", column_name, ":", len(res))
    return res, id2item, item2id

def max_value_in_column(dict, column_name):
    res = []
    for key in dict.keys():
        c_value = dict[key][column_name]
        res.append(c_value)
    max_res = max(res)
    min_res = min(res)
    return max_res, min_res


def main():
    print("PREPROCESSING")
    datapath = "./data/rectal ca_mri_20190917.xlsx"
    targetdir = "./results/"
    dl = DataLoader(datapath, targetdir)

    patients_records = dl.patients_parser(dl._targetdir + "patients.json", "list", _load_json=True)
    op_code_records = dl.op_parser(dl._targetdir + "op_name.json", "op_name", _load_json=True)
    op_code_binaries = unique_list_in_column(op_code_records, 'op_code')
    # print("UNIQUE OP CODE:", len(unique_op_code_list))

    # for research_no in op_name_records.keys():
    #     op_name_record = op_name_records[research_no]

    # survival_records = dl.survival_parser(dl._targetdir + "survival.json", "survival", _tsv_name="survival.tsv", _load_json=True, _load_tsv=True)
    # mri_records = dl.mri_parser(dl._targetdir + "mri.json", "MRI", _tsv_name="mri.tsv", _load_json=True)
    mri_records = dl.mri_parser(dl._targetdir + "mri.json", "MRI", filter_texts=['high risk', 'highrisk', 'High risk'], _load_json=True)

    cea_records = dl.cea_parser(dl._targetdir + 'cea.json', "CEA", _load_json=True)
    rt_records = dl.rt_parser(dl._targetdir + "rt.json", "RT", _load_json=True)
    # pathology_records = dl.pathology_parser(dl._targetdir + "pathology.json", "pathology", _tsv_name="pathology.tsv", _load_json=True)
    overall_survival_labels = generate_overall_survival_labels(patients_records, save_path="./results/overall_survival_labels.json")
    #generate_overall_survival_data_with_mri_cea_op_rt(overall_survival_labels, mri_records, patients_records, cea_records, op_code_records, op_code_binaries, rt_records, save_path="./results/overall_survival_data_with_mri_cea_op_rt.json")


    generate_overall_survival_data(
            overall_survival_labels,
            mri_records,
            patients_records,
            cea_records,
            op_code_records,
            op_code_binaries,
            rt_records,
            save_path="./results/overall_survival_data_with_mri_cea_op_rt.json")



if __name__ == '__main__':
    main()
