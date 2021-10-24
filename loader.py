import pandas as pd
import json
import sys, os
from datetime import date

import time

# Multiprocessing
# import parmap
import multiprocessing
from itertools import repeat
import pickle

def parse_patients_row(row, res):
    research_no = str(row[1]).strip()
    first_date = str(row[2]).strip()
    diag_code = str(row[3]).strip()
    diag_code_name = str(row[4]).strip()
    gender = str(row[5]).strip()
    birth_date = str(row[6]).strip()
    first_age = str(row[7]).strip()
    edt = str(row[10]).strip()
    last_date = str(row[11]).strip()
    survival = str(str(row[12]).strip() == "Y")
    sample = {
        'research_no': research_no,
        'first_date': first_date,
        'diag_code': diag_code,
        'diag_code_name': diag_code_name,
        'gender': gender,
        'birth_date': birth_date,
        'first_age': first_age,
        'edt': edt,
        'last_date': last_date,
        'survival': survival
    }

    if research_no in res.keys():
        res[research_no].append(sample)
    else:
        res[research_no] = [sample]

class DataLoader:
    def __init__(self, _inputpath, _targetdir):
        self._inputpath = _inputpath
        self._targetdir = _targetdir

        os.makedirs(self._targetdir, exist_ok=True)

        self.num_cores = multiprocessing.cpu_count()

        self.read_whole_file(_res_name="whole_file.pkl", _load_res=True)


    def read_whole_file(self, _res_name=None, _load_res=True):
        print("READING whole data:", self._inputpath)

        _res_path = self._targetdir + _res_name

        if _load_res:
            if _res_name is not None and os.path.exists(_res_path):
                with open(_res_path, 'rb') as rbf:
                    self.whole_file = pickle.load(rbf)
                    return

        self.whole_file = pd.read_excel(self._inputpath, sheet_name=None)

        if _res_name is not None:
            with open(_res_path, 'wb') as wbf:
                pickle.dump(self.whole_file, wbf)


    def read_raw_data(self, sheet, _tsv_name=None, _load_tsv=True):
        # Read excel sheet and convert to tsv

        print("READING", sheet, "sheet in excel...")

        _tsv_path = self._targetdir + _tsv_name

        if _load_tsv:
            # Load pre-read version
            if _tsv_name is not None and os.path.exists(_tsv_path):
                df = pd.read_csv(_tsv_path, sep='\t', index_col='Index')
                print(sheet, "loaded:", _tsv_path)
                return df

        start_time = time.time()
        df = pd.read_excel(self._inputpath, sheet_name=sheet)
        print(sheet, "in excel loaded:", df.shape, "in", (time.time() - start_time), "seconds")

        if _tsv_name is not None:
            df.to_csv(_tsv_path, sep='\t', index_label='Index')
        else:
            print("Couldn't save tsv for", sheet)

        return df

    def make_df_from_excel(self, sheet, _tsv_name=None, _load_tsv=True, nrows=10000):
        # Read excel sheet and convert to tsv

        print("READING", sheet, "sheet in excel...")

        _tsv_path = self._targetdir + _tsv_name

        if _load_tsv:
            if _tsv_path is not None and os.path.exists(_tsv_path):
                df = pd.read_csv(_tsv_path, sep='\t', index_col='Index')
                print(sheet, "loaded:", _tsv_path)
                return df

        start_time = time.time()
        df_header = pd.read_excel(self._inputpath, sheet_name=sheet, nrows=1)

        chunks = []
        i_chunk = 0
        skiprows = 1
        while True:
            df_chunk = pd.read_excel(self._inputpath, sheet_name=sheet, nrows=nrows, skiprows=skiprows, header=None)
            skiprows += nrows
            if not df_chunk.shape[0]:
                break
            else:
                print("  - chunk {i_chunk} ({df_chunk.shape[0]} rows)")
                chunks.append(df_chunk)
            i_chunk += 1

        df_chunks = pd.concat(chunks)
        columns = {i: col for i, col in enumerate(df_header.columns.tolist())}
        df_chunks.rename(columns=columns, inplace=True)
        df = pd.concat([df_header, df_chunks])

        print(sheet, "in excel loaded:", df.shape, "in", (time.time() - start_time), "seconds")

        if _tsv_path is not None:
            df.to_csv(_tsv_path, sep='\t', index_label='Index')
        else:
            print("Couldn't save tsv for", sheet)

        return df

    def remove_subtext(self, org_text, item):
        return ''.join(org_text.split(item))

    def datasaver(parse_func):
        def wrapper(*args, **kwargs):
            if kwargs['_load_json']:
                if os.path.exists(args[1]):
                    with open(args[1], 'r') as rf:
                        res = json.load(rf)
                    print(args[2], "(json) loaded with", args[1])
                    return res

            start_time = time.time()
            res = parse_func(*args, **kwargs)
            print("Generating", args[2], ":", (time.time() - start_time), "seconds")

            with open(args[1], 'w') as wf:
                json.dump(res, wf)

            return res

        return wrapper

    @datasaver
    def patients_parser(self, _jsonname, sheet, _load_json=True):
        # df = self.make_df_from_excel(sheet, _tsv_name, _load_tsv=_load_tsv)
        # df = self.read_raw_data(sheet, _tsv_name, _load_tsv = _load_tsv)
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}
        res = {}

        for row in df.itertuples():
            research_no = str(row[1]).strip()
            first_date = str(row[2]).strip()
            diag_code = str(row[3]).strip()
            diag_code_name = str(row[4]).strip()
            gender = str(row[5]).strip()
            birth_date = str(row[6]).strip()
            first_age = str(row[7]).strip()
            edt = str(row[10]).strip()
            last_date = str(row[11]).strip()
            survival = str(str(row[12]).strip() == "Y")
            sample = {
                'research_no': research_no,
                'first_date': first_date,
                'diag_code': diag_code,
                'diag_code_name': diag_code_name,
                'gender': gender,
                'birth_date': birth_date,
                'first_age': first_age,
                'edt': edt,
                'last_date': last_date,
                'survival': survival
            }

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]

        # pool = multiprocessing.Pool(self.num_cores)
        # manager = multiprocessing.Manager()
        # res = manager.dict()
        # row_list = [list(row) for row in df.itertuples()]
        #
        # pool.starmap(parse_patients_row, zip(row_list, repeat(res)))
        # pool.close()
        # pool.join()
        #
        # res = dict(res)
        # print("Total patients:", len(list(res.keys())))

        return res

    @datasaver
    def survival_parser(self, _jsonpath, sheet, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}
        res = {}

        for row in df.itertuples():
            research_no = str(row[1]).strip()
            icdocd = str(row[2]).strip()
            icdocd_name = str(row[3]).strip()
            survival = str(str(row[4]).strip() == "Y")
            edt = str(row[5]).strip()
            dc = str(row[6]).strip()
            dc_name = str(row[7]).strip()
            sample = {
                'research_no': research_no,
                'ICDOCd': icdocd,
                'ICDOCd_name': icdocd_name,
                'survival': survival,
                'edt': edt,
                'dc': dc,
                'dc_name': dc_name
            }
            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]

        return res

    @datasaver
    def mri_parser(self, _jsonpath, sheet, filter_texts = None, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}

        res = {}

        for row in df.itertuples():
            hospital_no = str(row[1]).strip()
            research_no = str(row[2]).strip()
            diag_code = str(row[3]).strip()
            diag_code_name = str(row[4]).strip()
            diag_date = str(row[5]).strip()
            conduct_date = str(row[6]).strip()
            reading_date = str(row[7]).strip()
            reading_text = str(row[8]).strip()
            if filter_texts:
                for f_t in filter_texts:
                    if f_t in reading_text:
                        # print(f_t)
                        # print(reading_text)
                        reading_text = self.remove_subtext(reading_text, f_t)
                        # print("_----")
                        # print(reading_text)
                        # raise
            form_no = str(row[9]).strip()
            gender = str(row[10]).strip()
            gender_name = str(row[11]).strip()
            birth_date = str(row[12]).strip()
            conduct_age = str(row[13]).strip()

            sample = {
                'hospital_no': hospital_no,
                'research_no': research_no,
                'diag_code': diag_code,
                'diag_code_name': diag_code_name,
                'diag_date': diag_date,
                'conduct_date': conduct_date,
                'reading_date': reading_date,
                'reading_text': reading_text,
                'form_no': form_no,
                'gender': gender,
                'gender_name': gender_name,
                'birth_date': birth_date,
                'conduct_age': conduct_age
            }

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]
        return res

    @datasaver
    def recur_parser(self, _jsonpath, sheet, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}

        res = {}
        for row in df.itertuples():
            research_no = str(row[2]).strip()
            diag_code_name = str(row[3]).strip()
            gender = str(row[4]).strip()
            first_date = str(row[5]).strip().replace(",", "-")
            birth_date = str(row[6]).strip()
            first_age = str(row[7]).strip()
            recur_date = str(row[8]).strip().replace(",", "-")
            recur_loc = str(row[9]).strip()
            last_date = str(row[10]).strip().replace(",", "-")
            edt = str(row[11]).strip().replace(",", "-")

            sample = {
                'research_no': research_no,
                'diag_code_name': diag_code_name,
                'gender': gender,
                'first_date': first_date,
                'birth_date': birth_date,
                'first_age': first_age,
                'recur_date': recur_date,
                'recur_loc': recur_loc,
                'last_date': last_date,
                'edt': edt
            }

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]
        return res

    @datasaver
    def pathology_parser(self, _jsonpath, sheet, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}

        res = {}
        for row in df.itertuples():
            research_no = str(row[1]).strip()
            diag_date = str(row[10]).strip()
            test_date = str(row[11]).strip()
            org_text = str(row[63]).strip()

            # print(research_no)
            # print(diag_date)
            # print(test_date)
            # print(org_text)
            # print('◇' in org_text)
            # org_text = '\n'.join(org_text.split('◇'))
            # print(org_text)
            #
            # raise

            sample = {
                'research_no': research_no,
                'diag_date': diag_date,
                'test_date': test_date,
                'org_text': org_text
            }

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]

        return res

    @datasaver
    def op_parser(self, _jsonpath, sheet, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}

        res = {}
        for row in df.itertuples():
            research_no = str(row[1]).strip()
            op_org = str(row[2]).strip()
            op_org_name = str(row[3]).strip()
            op_date = str(row[4]).strip()
            op_code = str(row[5]).strip()
            op_code_name = str(row[6]).strip()
            icd9cmcd = str(row[7]).strip()
            icd9cmcd_name = str(row[8]).strip()
            main_op = str(str(row[9]) == "Y") # ??

            sample = {
                'research_no': research_no,
                'op_org': op_org,
                'op_org_name': op_org_name,
                'op_date': op_date,
                'op_code': op_code,
                'op_code_name': op_code_name,
                'icd9cmcd': icd9cmcd,
                'icd9cmcd_name': icd9cmcd_name,
                'main_op': main_op
            }

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]
        return res

    @datasaver
    def rt_parser(self, _jsonpath, sheet, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}

        res = {}

        for row in df.itertuples():
            research_no = str(row[1]).strip()
            rt_date = str(row[5]).strip()
            rt_name = str(row[10]).strip()

            sample = {
                'research_no': research_no,
                'rt_date': rt_date,
                'rt_name': rt_name
            }

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]
        return res

    @datasaver
    def cea_parser(self, _jsonpath, sheet, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}

        res = {}

        for row in df.itertuples():
            research_no = str(row[1]).strip()
            cea_date = str(row[3]).strip().replace(",", "-")
            cea_code = str(row[5]).strip()
            cea_code_name = str(row[6]).strip()
            cea_unit_code = str(row[9]).strip()
            cea_unit_name = str(row[10]).strip()
            cea_unit_value = str(row[11]).strip()

            sample = {
                'research_no': research_no,
                'cea_date': cea_date,
                'cea_code': cea_code,
                'cea_code_name': cea_code_name,
                'cea_unit_code': cea_unit_code,
                'cea_unit_name': cea_unit_name,
                'cea_unit_value': cea_unit_value
            }

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]
        return res

    @datasaver
    def cap_parser(self, _jsonpath, sheet, _load_json=True):
        if sheet in self.whole_file.keys():
            df = self.whole_file[sheet]
        else:
            print("No sheet with name", sheet)
            return {}

        res = {}

        for row in df.itertuples():
            research_no = str(row[3]).strip()
            drug_code = str(row[10]).strip()
            drug_amount = str(row[12]).strip()
            cancer_group = str(row[24]).strip()
            start_date = str(row[34]).strip()
            end_date = str(row[35]).strip()
            _T = str(row[36]).strip()
            _N = str(row[37]).strip()
            _M = str(row[38]).strip()
            stage = str(row[39]).strip()
            weight = str(row[40]).strip()
            bsa = str(row[41]).strip()
            inject_type = str(row[42]).strip()
            ccr = str(row[45]).strip()
            gender = str(row[46]).strip()
            age = str(row[48]).strip()

            sample = {
                'research_no': research_no,
                'drug_code': drug_code,
                'drug_amount': drug_amount,
                'cancer_group': cancer_group,
                'start_date': start_date,
                'end_date': end_date,
                '_T': _T,
                '_N': _N,
                '_M': _M,
                'stage': stage,
                'weight': weight,
                'bsa': bsa,
                'inject_type': inject_type,
                'ccr': ccr,
                'gender': gender,
                'age': age
            }
            # print(sample)
            # raise

            if research_no in res.keys():
                res[research_no].append(sample)
            else:
                res[research_no] = [sample]



        return res


if __name__ == '__main__':
    # Test codes
    _inputpath = "./data/rectal ca_mri_20190917.xlsx"
    _targetdir = "./results/"
    dl = DataLoader(_inputpath, _targetdir)

    # start_time = time.time()
    # all_excel = pd.read_excel(_inputpath, sheet_name=None)
    # print("Reading whole file:", (time.time() - start_time), "seconds")
    # print('list' in all_excel.keys())
    # print(type(all_excel))
    # raise


    patients_records = dl.patients_parser(dl._targetdir + "patients.json", "list", _load_json=True)
    survival_records = dl.survival_parser(dl._targetdir + "survival.json", "survival", _load_json=True)
    mri_records = dl.mri_parser(dl._targetdir + "mri.json", "MRI", filter_texts=['high risk', 'highrisk', 'High risk'], _load_json=True)
    recur_records = dl.recur_parser(dl._targetdir + "recur.json", "recur", _load_json=True)
    op_name_records = dl.op_parser(dl._targetdir + "op_name.json", "op_name", _load_json=True)
    pathology_records = dl.pathology_parser(dl._targetdir + "pathology.json", "pathology", _load_json=True)
    cap_records = dl.cap_parser(dl._targetdir + "cap.json", "CAP", _load_json=True)
    rt_records = dl.rt_parser(dl._targetdir + "rt.json", "RT", _load_json=True)
