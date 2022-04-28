# Reference: https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm
HEALTHY_BMI_RANGE = [18.5, 24.9]

# Reference: https://www.bloodpressureuk.org/your-blood-pressure/understanding-your-blood-pressure/what-do-the-numbers-mean/#:~:text=%E2%80%A2%2090%2F60mmHg%20up%20to,it%20in%20the%20healthy%20range
HEALTHY_SBP_RANGE = [95, 120]
HEALTHY_DBP_RANGE = [65, 80]
SBP_DBP_DIFF_RANGE = [30, 40]

# Reference: https://pubmed.ncbi.nlm.nih.gov/16546483/
HEALTHY_BSA_RANGE = [1.62, 2]

DB_INFO = {
    'shareedb': {
        'annotations': 'qrs',
        'user_info_filename': 'info.txt',
    },
    'nsrdb': {
        'annotations': 'atr',
        'men': 5,
        'women': 13,
        'age_range_men': [26, 45],
        'age_range_women': [20, 50],
    },
    'nsr2db': {
        'annotations': 'ecg',
        'men': 30,
        'women': 24,
        'age_range_men': [28, 76],
        'age_range_women': [58, 73],
    }
}
