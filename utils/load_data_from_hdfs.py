# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/5/15 13:13
import os


def load_data_from_hdfs(part_num, source_file, target_file):
    str_list = ["hadoop fs -get /user/recsys/qihengda/", source_file, "/", part_num, ' ' + target_file + "/"]
    cmd = ''.join(str_list)
    print(cmd)
    os.system(cmd)


def gen_part_num(part_num_file):
    cmd = "hadoop fs -ls /user/recsys/qihengda/train.gender.libsvm > " + part_num_file
    os.system(cmd)
    lines = open(part_num_file, "r").readlines()
    part_nums = []
    for line in lines[1:3]:
        part_nums.append(line.split(" ")[-1].split("/")[-1])
    return part_nums


def main():
    part_nums = gen_part_num("part_nums_file")
    for part_num in part_nums:
        load_data_from_hdfs(part_num, "train.gender.libsvm", "gender_train_data")


if __name__ == '__main__':
    main()
