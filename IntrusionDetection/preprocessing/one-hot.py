# coding:utf-8
# 离散型特征one-hot化处理

from numpy import argmax
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import csv
"""
back,buffer_overflow,ftp_write,guess_passwd,imap,ipsweep,land,loadmodule,multihop,neptune,nmap,normal,perl,phf,pod,portsweep,rootkit,satan,smurf,spy,teardrop,warezclient,warezmaster.
duration: continuous.
protocol_type: symbolic.
service: symbolic.
flag: symbolic.
src_bytes: continuous.
dst_bytes: continuous.
land: symbolic.
wrong_fragment: continuous.
urgent: continuous.
hot: continuous.
num_failed_logins: continuous.
logged_in: symbolic.
num_compromised: continuous.
root_shell: continuous.(potential symbolic)
su_attempted: continuous.(potential symbolic)
num_root: continuous.
num_file_creations: continuous.
num_shells: continuous.
num_access_files: continuous.
num_outbound_cmds: continuous.
is_host_login: symbolic.
is_guest_login: symbolic.
count: continuous.
srv_count: continuous.
serror_rate: continuous.
srv_serror_rate: continuous.
rerror_rate: continuous.
srv_rerror_rate: continuous.
same_srv_rate: continuous.
diff_srv_rate: continuous.
srv_diff_host_rate: continuous.
dst_host_count: continuous.
dst_host_srv_count: continuous.
dst_host_same_srv_rate: continuous.
dst_host_diff_srv_rate: continuous.
dst_host_same_src_port_rate: continuous.
dst_host_srv_diff_host_rate: continuous.
dst_host_serror_rate: continuous.
dst_host_srv_serror_rate: continuous.
dst_host_rerror_rate: continuous.
dst_host_srv_rerror_rate: continuous.
"""


def one_hot(df):
    protocol_type_one_hot = pd.get_dummies(df["protocol_type"],prefix="protocol_type")
    df = df.drop("protocol_type", axis=1)
    df = df.join(protocol_type_one_hot)

    service_one_hot = pd.get_dummies(df["service"],prefix="service")
    df = df.drop("service", axis=1)
    df = df.join(service_one_hot)

    flag_one_hot = pd.get_dummies(df["flag"],prefix="flag")
    df = df.drop("flag", axis=1)
    df = df.join(flag_one_hot)

    land_one_hot = pd.get_dummies(df["land"],prefix="land")
    df = df.drop("land", axis=1)
    df = df.join(land_one_hot)

    logged_in_one_hot = pd.get_dummies(df["logged_in"], prefix="logged_in")
    df = df.drop("logged_in", axis=1)
    df = df.join(logged_in_one_hot)

    # root_shell_one_hot = pd.get_dummies(df["root_shell"], prefix="root_shell")
    # df = df.drop("root_shell", axis=1)
    # df = df.join(root_shell_one_hot)
    #
    # su_attempted_one_hot = pd.get_dummies(df["su_attempted"], prefix="su_attempted")
    # df = df.drop("su_attempted", axis=1)
    # df = df.join(su_attempted_one_hot)

    is_host_login_one_hot = pd.get_dummies(df["is_host_login"], prefix="is_host_login")
    df = df.drop("is_host_login", axis=1)
    df = df.join(is_host_login_one_hot)

    is_guest_login_one_hot = pd.get_dummies(df["is_guest_login"], prefix="is_guest_login")
    df = df.drop("is_guest_login", axis=1)
    df = df.join(is_guest_login_one_hot)
    return df


if __name__ == '__main__':
    df = pd.read_csv('../dataset/KDDCUP99/processed/kddcup.data_10_percent_corrected.csv')
    # print(df)
    # df = one_hot(df,"protocol_type")
    # df = one_hot(df, "service")
    # df = one_hot(df, "flag")
    # df = one_hot(df, "land")
    # df = one_hot(df, "logged_in")
    # df = one_hot(df, "is_host_login")
    # df = one_hot(df, "is_guest_login")
    df = one_hot(df)
    print(df)
    dataset = df.to_csv('../dataset/KDDCUP99/processed/kddcup_10_percent_onehot.csv',header=True,index=0)
    ds = pd.read_csv('../dataset/KDDCUP99/processed/kddcup_10_percent_onehot_ordered.csv')
    print(ds)

