import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import metrics

#World Governance Indicators, to look at change in indicators over time
wgi = pd.read_csv('../data/transformed/wgi_pivot.csv')

#import cleaned FIW data
fiw = pd.read_csv('../data/transformed/fiw_clean.csv')

#fully cleaned and merged df for general exploration
df = pd.read_csv('../data/transformed/mm_wgi_fiw.csv')

#importing cleaned mm data
mm = pd.read_csv('../data/transformed/mass_mobilization_data_cleaned.csv')

# encoding categorical columns prior to modeling.

def categorical_to_encode(df,  columns_to_label_encode, columns_to_get_dummies):
    '''
    function to apply LabelEncoder and/or GetDummies to prepare dataframe for modeling.
    Inputs:
    df = a dataframe
    columns_to_label_encode: a list of columns to apply LabelEncoder
    columns_to_get_dummies: a list of columns to apply GetDummies
    Output:
    an encoded data frame ready for modeling.
    '''
    #resetting 'duration' column to remove letters, resulting in a numeric column

    for column in columns_to_label_encode:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    for column in columns_to_get_dummies:
        dummy_df = pd.get_dummies(df[column], drop_first=True)
        df = pd.concat([df, dummy_df], axis=1)

    df.drop(columns = columns_to_get_dummies, inplace=True)

    return df

def db_country_time_series(df, country):
    '''
    input: a cleaned worldbank dataframe and a country
    output: a country specific dataframe with a datetime index by year
    '''
    #filter dataframe by country
    df = df[df['country_name'] ==  country]
    #sort by year
    df = df.sort_values(by='year')
    #set index to datetime object by year
    df.index = pd.to_datetime(df['year'], format='%Y').dt.year
    #return datetime index dataframe for specified country
    return df


def time_series_by_country(df, place, agg_method = 'sum', interval = 'MS'):

    '''
    A function to downsample the full dataframe to view data by country.
    Date range will include all years for which data is available for a specified country.
    Input:
        * a dataframe - defaults to 'df'
        * place - country or region name
        * agg_method - aggregation method for resampling. defaults to 'sum'. 'mean' also recommended.
        * interval - 'YS': annual / 'MS': monthly. Default is monthly.

    '''
    try:
        # generate downsamples dataframe based on country selection
        output_df = df[df['country_name'] == place.title()]

        #set index to date_time based on protest startdate
        output_df = output_df.sort_values(by='start_date')
        output_df.set_index(pd.DatetimeIndex(output_df['start_date']), inplace=True)

        #specify date range if desired
        start_date = str(output_df.index.min())[:4]
        end_date = str(output_df.index.max())[:4]
        output_df = output_df.loc[start_date:end_date]

        output_df.drop(columns = ['year', 'end_date', 'ccode', 'country_name', 'start_date', 'duration'], inplace=True)

        # resample to look at data based on regular intervals
        output_df.resample(interval).protest.agg(agg_method)


        return output_df

    except:

        print(f'We don\'t have enough data to fulfill your request for {place}. Please check your spelling, or try another location.')

def model_metrics(some_lr, scaled=False):

    if scaled:
        train_r2 = some_lr.score(X_train_sc, y_train)
        test_r2 = some_lr.score(X_test_sc, y_test)
    else:

        train_r2 = some_lr.score(X_train, y_train)
        test_r2 = some_lr.score(X_test, y_test)
    preds = some_lr.predict(X)
    resids_mean = (y - preds).mean()
    mae = metrics.mean_absolute_error(y, preds)
    resids = resids = y - preds
    rss = (resids ** 2).sum()
    mse = metrics.mean_squared_error(y, preds)
    rmse = np.sqrt(metrics.mean_squared_error(y, preds))
    cvs = cross_val_score(some_lr, X, y, cv=5).mean()

    final_dict={'Train R2 Score': train_r2,
                'Test R2 Score' : test_r2,
                'Mean of Residuals': resids_mean,
               'Mean Absolute Error': mae,
               'Residual Sum of Squares': rss,
               'Mean Squared Error': mse,
               'Root Mean Squared Error': rmse,
               'cross_val_score': cvs}
    return final_dict


# Cleaning Functions
def clean_mass_mob_df(df):
    '''
    Clean Mass Mobilization dataset
    input: mass mobilization df
    output: cleaned mass mobilization df
    '''
    #Drop nulls in startday/month/year and endday/month/year:
    df = df.drop(df[df["startyear"].isnull()].index)
    df.reset_index(inplace=True)


    #set startyear/month/day and endday/month/year to int, then string, in preparation for concatenation
    df = df.astype({"startyear": int, "startmonth": int, 'startday': int, "endyear": int, "endmonth": int, 'endday': int})
    df = df.astype({"startyear": str, "startmonth": str, 'startday': str, "endyear": str, "endmonth": str, 'endday': str})

    #create start date and end date columns and set as datetime object
    df['start_date'] = df['startyear'] + '-' + df['startmonth'] + '-'+ df['startday']
    df['end_date'] = df['endyear'] + '-' + df['endmonth'] + '-'+ df['endday']
    df['start_date'] = pd.to_datetime(df['start_date'], yearfirst=True)
    df['end_date'] = pd.to_datetime(df['end_date'], yearfirst=True)

    #setting df to only include dates between 2006 and 2020, in order to align with wgi and fiw data.
    df = df[(df['start_date']>'2006') & (df['start_date']<'2021')]

    #engineer a column to give duration of protest
    df['duration'] = df['end_date'] - df['start_date'] + timedelta(days=1)
    #reset duration column as an int
    df['duration_int'] = df['duration'].dt.days

    # converting protester violence from float to int, and filling nulls with 0
    df['protesterviolence'].fillna(0.0, inplace=True)
    df['protesterviolence'] = df['protesterviolence'].astype(int)


    #merge participants and participants category
    map_list = []
    for row in df['participants']:
        try:
            map_list.append(row.strip('s><+ abcdefghijklmnopqrstuvwxyz!@#$%^&*():";."').split('-', 1)[0])
        except:
            map_list.append(row)

    df['participants'] = map_list
    #casting participants as numeric
    pd.to_numeric(df['participants'], errors='coerce').fillna(0, inplace=True)

    # setting participants to fit into participants_category ranges
    cat_map_list = []
    for row in df['participants']:
        try:
            if int(row) > 1 and int(row) < 100:
                cat_map_list.append('50-99')
            elif int(row) > 99 and int(row) < 1000:
                cat_map_list.append('100-999')
            elif int(row) > 999 and int(row) < 2000:
                cat_map_list.append('1000-1999')
            elif int(row) > 2000 and int(row) < 5000:
                cat_map_list.append('2000-4999')
            elif int(row) > 4999 and int(row) < 9999:
                cat_map_list.append('5000-10000')
            elif int(row) > 10000:
                cat_map_list.append('>10000')
            else:
                cat_map_list.append('unknown')
        except:
            cat_map_list.append('unknown')

    df['participants'] = cat_map_list

    # mapping participants_category ranges to numerical category
    participants_category_map = {
        'unknown': 1,
        '50-99': 2,
        '100-999': 3,
        '2000-4999': 4,
        '1000-1999': 5,
        '5000-10000': 6,
        '>10000': 7
    }

    df['participants_category'].fillna('NaN', inplace=True)
    df['participants_category'] = df.apply(f, axis=1)
    df['participants_category'] = df['participants_category'].map(participants_category_map)
    #fill nulls with 'unknown'
    df.fillna('unknown', inplace=True)

    #resetting index
    df.reset_index(drop=True,inplace=True)

    #replacing country names in mm to align with wgi country names

    country_changes_mm = {'Macedonia': 'North Macedonia',
                          'Bosnia':'Bosnia and Herzegovina',
                          'Yugoslavia':'Serbia',
                          'Serbia and Montenegro': 'Montenegro',
                          'Russia': 'Russian Federation',
                          'Cape Verde': 'Cabo Verde',
                          'Gambia': 'Gambia, The',
                          'Ivory Coast': 'Cote d\'Ivoire',
                          'Congo Brazzaville': 'Congo, Rep.',
                          'Congo Kinshasa':'Congo, Dem. Rep.',
                          'Swaziland': 'Eswatini',
                          'United Arab Emirate': 'United Arab Emirates',
                          'Timor Leste':'Timor-Leste'}


    for name in df['country']:
        for k, v in country_changes_mm.items():
            if name == k:
                df.replace({name: v}, inplace=True)



    #renaming country column to country_name to align with wgi dataframe for merging
    df = df.rename({'country':'country_name'}, axis=1)

    # binarizing protester demand columns
    demands_df = df[['protesterdemand1', 'protesterdemand2', 'protesterdemand3', 'protesterdemand4']]
    demands_df = demands_df.stack().str.get_dummies().sum(level=0).drop(columns=['.', 'unknown'])
    df = pd.concat([df, demands_df], axis=1)

    #drop original date columns
    df.drop(columns = ['startday', 'startmonth', 'startyear', 'endday', 'endmonth', 'endyear',
                       'index', 'location', 'sources', 'notes', 'participants',
                      'protesteridentity', 'protesterdemand1', 'protesterdemand2', 'protesterdemand3',
                       'protesterdemand4', 'stateresponse2', 'stateresponse3', 'stateresponse4',
                       'stateresponse5', 'stateresponse6', 'stateresponse7', 'id'], inplace=True)

    return df


def clean_fiw_for_model(df):
    '''
    Clean Freedom in the World dataset prior to merging with Mass Mobilization
    '''
    #preparing to drop all but my selected columns
    columns_to_drop = df.drop(columns=['Country/Territory', 'Status', 'Edition']).columns

    df.drop(columns=columns_to_drop, inplace=True)

    #renaming columns to match mm and wgi datasets to prepare for merging
    fiw_column_dict = {}
    for column in df.columns:
        if column == 'Country/Territory':
            fiw_column_dict[column] = 'country_name'
        elif column == 'Edition':
            fiw_column_dict[column] = 'year'
        else:
            fiw_column_dict[column] = 'fiw_' + column.lower().replace(' ', '_')

    df.rename(mapper=fiw_column_dict, axis=1, inplace=True)


    #replacing country names in mm to align with wgi country names

    country_changes_fiw = {'Slovakia': 'Slovak Republic',
                          'Serbia and Montenegro':'Serbia',
                          'Russia': 'Russian Federation',
                          'The Gambia': 'Gambia, The',
                          'Congo (Brazzaville)': 'Congo, Rep.',
                          'Congo (Kinshasa)':'Congo, Dem. Rep.'

}

    for name in df['country_name']:
        for k, v in country_changes_fiw.items():
            if name == k:
                df.replace({name: v}, inplace=True)

    #setting year column to a datetime object
    df['year'] = pd.to_datetime(df['year'], format='%Y', utc=True).dt.year

    return df


def clean_databank(df, csv_name):
    '''
    clean WorldBank databank dataset
    input: csv from worldbank databank and the name for output csv
    output: pivot table and csv
    In order to merge the output csv with the mass mobilization dataset, reset the columns to be:
    year, country, and one column per scoring metric by metric name.
    '''
    #dropping country code column
    df.drop(columns=['Country Code', 'Series Name'], inplace=True)

    #rename year columns
    for column in df.columns[2:]:
        df.rename(columns = {column: column[:4]}, inplace=True)

    for column in df.columns[:2]:
        df.rename(columns = {column: column.lower().replace(' ', '_')}, inplace=True)

    #create columns for 1997, 1999, and 2001, which will replicate the previous year, so that I have data for each year for analysis purposes.
    #However, I will only use 2006-2021 data in modeling, for use with the mm and fiw data.
    df['1997'] = df['1996']
    df['1999'] = df['1998']
    df['2001'] = df['2000']

    #reorder columns
    df = df[['country_name', 'series_code', '1996', '1997', '1998', '1999', '2000',
       '2001','2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
       '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
       '2020']]

    #dropping unneeded series codes
    db_series = ['CC.EST', 'GE.EST', 'PV.NO.SRC', 'RL.EST', 'VA.EST', 'SI.POV.GINI', 'SP.POP.TOTL', 'SI.SPR.BL50.ZS', 'SI.POV.ATTM.MI',
                 'SI.POV.ENRL.MI', 'SI.POV.WATR.MI', 'SI.POV.ELEC.MI', 'SI.DST.05TH.20', 'NY.GDP.PCAP.CD', 'NY.GDP.PCAP.KD.ZG']
    df = df[df['series_code'].isin(db_series)]

    #replacing country names in wgi to align with mm
    country_changes_db = {'Venezuela, RB': 'Venezuela',
                          'Iran, Islamic Rep.': 'Iran',
                          'Egypt, Arab Rep.': 'Egypt',
                          'Syrian Arab Republic': 'Syria',
                          'Yemen, Rep.': 'Yemen',
                          'Kyrgyz Republic': 'Kyrgyzstan',
                          'Taiwan, China': 'Taiwan',
                          'Korea, Dem. People\'s Rep.': 'North Korea',
                          'Korea, Rep.': 'South Korea'}

    for name in df['country_name']:
        for k, v in country_changes_db.items():
            if name == k:
                df.replace({name: v}, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # using melt to get rows and country_name set for merging with other dataframes
    # referenced stackoverflow for melt documentation: https://stackoverflow.com/questions/28654047/convert-columns-into-rows-with-pandas
    df = df.melt(id_vars=['country_name', 'series_code'])
    df.rename(columns = {'variable': 'year', 'value': 'score'}, inplace=True)

    # setting year as datetime object
    df['year'] = pd.to_datetime(df['year'], format='%Y').dt.year

    #converting to a pivot table to get the format needed to merge with mm.
    df_pivot = df.pivot(index=['country_name','year'], columns=['series_code'], values=['score'])
    df = pd.DataFrame(df_pivot)

    #filling fields containing '..' with NaNs.
    df.replace('..',np.NaN, inplace=True)


    #I save this cleaned data as a csv, then open the csv in excel to reset the headers, before reading the data back in to merge with mm and fiw.
    df.to_csv(f'./data/transformed/{csv_name}_pivot.csv')

    return df

def time_series_by_country(df=wgi, df2=fiw, df3 = mm, country=None, cols=['CC.EST', 'GE.EST', 'PV.NO.SRC', 'RL.EST', 'VA.EST'],
                           cols2=['fiw_status'], cols3=['protest'],title='Title', xlab='Year', ylab='Metric Score / Protest Count', steps=1):

        '''
        A function to plot WGI, protest count, and FIW score by year.
        input: a cleaned worldbank dataframe and a country
        output: a country specific dataframe with a datetime index by year and plot it
        '''
        #filter wgi dataframe by country and plot
        df = df[df['country_name'] ==  country.title()]
        #sort by year
        df = df.sort_values(by='year')
        #set index to datetime object by year
        df.index = pd.to_datetime(df['year'], format='%Y').dt.year
        #return datetime index dataframe for specified country


        #filter fiw dataframe by country and plot
        le = LabelEncoder()
        df2['fiw_status'] = le.fit_transform(df2['fiw_status'])
        df2 = df2[df2['country_name'] ==  country.title()]
        #sort by year
        df2 = df2.sort_values(by='year')
        #set index to datetime object by year
        df2.index = pd.to_datetime(df2['year'], format='%Y').dt.year
        #return datetime index dataframe for specified country

        #clean and filter mm dataframe by country/date and plot
        cols_to_drop = df3.drop(columns = ['country_name', 'year', 'region', 'protest', 'duration_int', 'region']).columns
        df3 = df3.drop(cols_to_drop, axis=1)
        df3 = df3[df3['country_name'] == country.title()]
        df3 = pd.DataFrame(df3.groupby(df3['year'])['protest'].sum())
        df3.index = pd.to_datetime(df3.index, format='%Y').year

        plt.figure(figsize=(18,9))

        # Iterate through each column name.
        for col in cols:

            # Generate a line plot of the column name.
            # You only have to specify Y, since our
            # index will be a datetime index.
            plt.plot(df[col], label=col)

        for col2 in cols2:

            # Generate a line plot of the column name.
            # You only have to specify Y, since our
            # index will be a datetime index.
            plt.plot(df2[col2], label=col2, marker='^')

        for col3 in cols3:

            # Generate a line plot of the column name.
            # You only have to specify Y, since our
            # index will be a datetime index.
            plt.plot(df3[col3], label=col3, marker='o')



        # Generate title and labels.
        plt.title(f'World Governance Indicators, Freedom Score, and Protest Count from 1996-2021: {country.title()}', fontsize=26)
        plt.xlabel(xlab, fontsize=20)
        plt.ylabel(ylab, fontsize=20)
        plt.legend()

        # Enlarge tick marks.
        plt.yticks(fontsize=18)
        plt.xticks(df.index[0::steps], fontsize=12, rotation=20);
