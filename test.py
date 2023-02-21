import os
import xml.etree.ElementTree as ET
import json
import pandas as pd
from datetime import datetime
import csv


def get_data(_get):
    try:
        return _get
    except:
        return ""


def get_keys_1(_data, _list):
    _temp = ""

    return _temp


def get_keys_2(_data, _list):
    _temp = []
    for i in _list:
        try:
            if i == "path":
                temp = "https://www.zu.ac.ae/main" + get_data(_data[i])
            else:
                temp = get_data(_data[i])

            if i == "created-on" or i == "last-modified" or i == "last-published-on":
               timestamp =  get_data(_data[i])[:10]
               temp =  datetime.fromtimestamp(int(timestamp))

            _temp.append(temp)

        except:
            temp = "NA"
            _temp.append(temp)

    return _temp


def write_json_data(_data, _writer):
    try:
        # _list_1 = ['ServiceName', 'GeneratedLink', 'Description', 'Procedures', "title",
        #            'ServiceTime', 'ServiceLocation', 'ServiceContact', 'SDGoals']
        _list_1 = ['ServiceName', 'ServiceCode', 'GeneratedLink', 'Description', "Procedures",
                   'TargetAudience', 'ServiceTime', 'ServiceFee', 'ServiceLocation', 'ServiceContact', 'SDGoals', 'title', 'name',
                   'path', 'keywords', 'ServiceUrl', 'IconFileUrl', 'DescriptionTrimmed', 'ServiceID']
        _flag = _data[_list_1[0]]
        csv_data = get_keys_2(_data, _list_1)

    except:
        _list_2 = ['path', 'name', 'created-on','title', 'description', 'last-published-on', 'last-modified']

                #    ['name', 'title', 'path', 'description',
                #    'last-published-on', 'created-on', 'last-modified']

        csv_data = get_keys_2(_data, _list_2)

    _writer.writerow(csv_data)


def get_xml_data(_data, _list):
    temp = []
    for i in _list:
        try:
            if i == "created-on" or i == "last-modified":
                _temp = get_data(datetime.fromtimestamp(int(_data.find(i).text)))
            
            elif i == "path":
                _temp = "https://www.zu.ac.ae/main" + get_data(_data.find(i).text)

            else:
                _temp = get_data(_data.find(i).text)

            temp.append(_temp)
        except:
            pass

    return temp


def write_file_data(filename):
    if ".xml" in filename:
        txt_file_sys_folder = "ZU_all_files_to_csv" + os.sep + \
            filename.split("\\")[-1][:-4] + "_system_folder" + ".csv"
        txt_file_sys_page = "ZU_all_files_to_csv" + os.sep + \
            filename.split("\\")[-1][:-4] + "_system_page" + ".csv"
        txt_file_sys_file = "ZU_all_files_to_csv" + os.sep + \
            filename.split("\\")[-1][:-4] + "_system_file" + ".csv"

        tree = ET.parse(filename)
        root = tree.getroot()

        sys_folder = [elem for elem in root.iter("system-folder")]
        sys_page = [elem for elem in root.iter("system-page")]
        sys_file = [elem for elem in root.iter("system-file")]

        sys_folder_list = []
        for child in sys_folder:
            # print("child", child.find('name').text)
            try:
                sys_folder_list.append(
                    ["https://www.zu.ac.ae/main" + child.find('path').text, child.find('name').text,
                    datetime.fromtimestamp(int(child.find('created-on').text)),
                    datetime.fromtimestamp(int(child.find('last-modified').text))])
            except Exception as e:
                print("In Exception", e)
                sys_folder_list.append(
                    ["https://www.zu.ac.ae/main" + child.find('path').text, child.find('title').text,
                    datetime.fromtimestamp(int(child.find('created-on').text)),
                    datetime.fromtimestamp(int(child.find('last-modified').text))])

        if len(sys_folder_list) > 0:
            with open(txt_file_sys_folder, "w", encoding="utf-8", newline="") as fp_folder:
                csv_writer = csv.writer(fp_folder)
                csv_writer.writerow(
                    ['path', 'name', 'created_on', 'last_modified'])
                csv_writer.writerows(sys_folder_list)

        sys_page_list = []
        for child in sys_page:
            try:
                _list = ['path', 'title', 'summary', 'created-on', 'last-modified']
                sys_page_list.append(get_xml_data(child, _list))
            except:
                _list = ['path', 'name', 'summary', 'created-on', 'last-modified']
                sys_page_list.append(get_xml_data(child, _list))

        if len(sys_page_list) > 0:
            with open(txt_file_sys_page, "w", encoding="utf-8", newline="") as fp_folder:
                csv_writer = csv.writer(fp_folder)
                csv_writer.writerow(
                    ['path','title','summary', 'created-on', 'last-modified'])
                csv_writer.writerows(sys_page_list)

        sys_file_list = []
        for child in sys_file:
            try:
                sys_file_list.append(
                    ["https://www.zu.ac.ae/main" + child.find('path').text,child.find('title').text,
                    datetime.fromtimestamp(int(child.find('created-on').text[:10])),
                    datetime.fromtimestamp(int(child.find('last-modified').text[:10]))])
            except:
                sys_file_list.append(
                    ["https://www.zu.ac.ae/main" + child.find('path').text, child.find('name').text,
                    datetime.fromtimestamp(int(child.find('created-on').text[:10])),
                    datetime.fromtimestamp(int(child.find('last-modified').text[:10]))])

        if len(sys_file_list) > 0:
            with open(txt_file_sys_file, "w", encoding="utf-8", newline="") as fp_folder:
                csv_writer = csv.writer(fp_folder)
                csv_writer.writerow(
                    ['path', 'name', 'created-on', 'last-modified'])
                csv_writer.writerows(sys_file_list)

    elif ".json" in filename:
        txt_file = "ZU_all_files_to_csv" + os.sep + filename.split("\\")[-1][:-5] + ".csv"
        with open(filename, encoding='utf-8') as f:
            data = json.load(f)

            fp = open(txt_file, "w", encoding="utf-8", newline="")
            csv_writer = csv.writer(fp)

            try:
                _data = data['assets']
                # _list_2 = ['name', 'title', 'path', 'description',
                #            'last-published-on', 'created-on', 'last-modified']
                _list_2 = ['path', 'name', 'created-on','title', 'description', 'last-published-on', 'last-modified']
                csv_writer.writerow(_list_2)
                for i in _data:
                    write_json_data(i, csv_writer)

            except:
                _list_1 = ['ServiceName', 'ServiceCode', 'GeneratedLink', 'Description', "Procedures",
                   'TargetAudience', 'ServiceTime', 'ServiceFee', 'ServiceLocation', 'ServiceContact', 'SDGoals', 'title', 'name',
                   'path', 'keywords', 'ServiceUrl', 'IconFileUrl', 'DescriptionTrimmed', 'ServiceID']
                csv_writer.writerow(_list_1)
                for i in data:
                    write_json_data(i, csv_writer)

            fp.close()


def get_structured_data(_sys_, _main_list):
    for j in _sys_:
        name = j[0]
        # if i.upper().strip() in name.upper().strip() or i.upper().strip() == name.upper().strip():
        _main_list.append(j)

    return _main_list


def list_to_str(_list):
    _str = ""
    if len(_list) > 1:
        for i in _list:
            if i == _list[-1]:
                _str += i

            else:
                _str += i + " "

    return _str


for root, dir, files in os.walk(f'ZU_New_Files'):
    for _file in files:
        print("INPUT FILE", (os.path.join(root, _file)))
        write_file_data(os.path.join(root, _file))

        # if data[0] == "xml":
        #     sys_folder = data[1]
        #     sys_page = data[2]
        #     sys_file = data[3]

        #     all_xml = []

        #     all_xml = get_structured_data(sys_folder, all_xml)
        #     all_xml = get_structured_data(sys_page, all_xml)
        #     all_xml = get_structured_data(sys_file, all_xml)

        #     links_ratio = []
        #     for i in all_xml:
        #         links_ratio.append([i[0], i[1][0], i[1][1], i[1][2]])

        #     df1 = pd.DataFrame(links_ratio, columns=[
        #                         'single_ratio', 'actual_ratio', 'name', 'path', 'timestamp'])
        #     main_df = main_df.append(df1, ignore_index=True)

        # if data[0] == "json":
        #     json_data = data[1]
        #     print(len(json_data))
        #     print(json_data[1])
        #     print(json_data[2])
        #     print(json_data[3])

        #     all_json = []
        #     all_json = get_structured_data(json_data, all_json)

        #     links_ratio = []
        #     # for i in all_json:
        #     #     links_ratio.append([i[0], i[1][0], i[1][1], i[1][2]])

        #     # df1 = pd.DataFrame(links_ratio, columns=[
        #     #                     'single_ratio', 'actual_ratio', 'name', 'path', 'timestamp'])
        #     # main_df = main_df.append(df1, ignore_index=True)
