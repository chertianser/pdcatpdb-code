import os
import numpy as np
import glob

Wavelength = [210, 230, 250, 280, 330]

def find_Sig_start(report, Sig_Num = 1):
    Sig_chan = 'Signal ' + str (Sig_Num)
    for i, line in enumerate(report):
        if line.startswith(Sig_chan):
            return i
        
def find_Sig_end(report, Sig_Num = 1):
    end_list = []
    for i, line in enumerate(report):
        if line.startswith('Totals :'):
            end_list.append(i)
    return end_list[Sig_Num - 1]

def pickline(text, Lines):
    return [line for i, line in enumerate(text) if i in Lines]


def single_line_result(Lines):
    Num_Lines = len(Lines)
    Results = np.empty([Num_Lines,4])
    for i, line in enumerate(Lines):
        Results[i,0] = float(line[7:12])
        Results[i,1] = float(line[18:25])
        Results[i,2] = float(line[26:36])
        Results[i,3] = float(line[37:47])
    return Results

def get_all_peaks(report, Sig_Num = 1):
    with open(report,'r',encoding='utf-16 le') as f1:
        try:
            Sig_Start = find_Sig_start(f1, Sig_Num) + 5
        except:
            print('Signal not found!')
            return np.zeros((1,4))
    with open(report,'r',encoding='utf-16 le') as f2:
        Sig_End = find_Sig_end(f2, Sig_Num)
    Line_list = [] 
    for i in range (Sig_Start, Sig_End):
        Line_list.append(i)
    with open(report,'r',encoding='utf-16 le') as f:
        Sig_All = pickline(f, Line_list)
    Peak_Results = single_line_result(Sig_All)
    return Peak_Results
    
def match_peak(Peak_Results, RetTime = 1, TimeError = 0.025):
    Peak_Found = False
    for i in range (0, Peak_Results.shape[0]):
        if ((1 - TimeError) * RetTime < Peak_Results[i,0] < (1 + TimeError) * RetTime):
            Peak_Found = True
            return Peak_Results[i,:]
    if not Peak_Found:
        return np.array([RetTime,0,0,0])
        
def gen_CSV():
    print(f'Please input Signal Number to search\n1 = {Wavelength[0]} nm\t2 = {Wavelength[1]} nm\t3 = {Wavelength[2]} nm\t4 = {Wavelength[3]} nm\t5 = {Wavelength[4]} nm')
    Sig_Num = int(input())
    print('1: single RetTime\t2: all RetTime')
    if input() == '1':
        print('Please input retention time to search:')
        RetTime = float(input())
        print('Please input error of RetTime')
        TimeError = float(input())/100
        To_Write = read_TXT_report(RetTime = RetTime, TimeError = TimeError, Sig_Num = Sig_Num, All_Peaks = False)
        CSV_Name = f'{Wavelength[Sig_Num-1]}_nm_' + str('{:.3f}'.format(RetTime)) + '_min.csv'
    np.savetxt(CSV_Name, To_Write, fmt = '%.6e', delimiter = ',', header = 'RetTime,Width,Area,Height', comments = '')

def read_TXT_report(RetTime, TimeError, Sig_Num, All_Peaks):
    Result_Array = np.empty((0,4))
    for Folder_Path in glob.glob((os.getcwd() +'\\???-*.D')):
        if 'no Sample Name' in Folder_Path:
            pass
        else:
            Report_Path = Folder_Path + '\\report.TXT'
            Peak_Results = get_all_peaks(Report_Path, Sig_Num = Sig_Num)
            if All_Peaks:
                Result_Array = np.vstack((Result_Array,Peak_Results))
            else:
                Peak_Matched = match_peak(Peak_Results, RetTime = RetTime, TimeError = TimeError)
                Result_Array = np.vstack((Result_Array,Peak_Matched))
    return Result_Array

def comb_all_peaks():
    pass


while True:
    gen_CSV()
    print('Continue? Y/N:')
    if ('n' in input()) or ('N' in input()):
        break
    else:
        pass