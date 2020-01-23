import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.interpolate import CubicSpline



weeks_in_month = (365/7)*(1/12)
maturity_rub = [3,6,9,12,24,36,48,60,72,84,96,108,120,144,180,]
maturity_rub= [i * weeks_in_month for i in maturity_rub]

ds_rate_rub = [
    8.51987757,
    8.68611319,
    8.90460383,
    9.04938647,
    9.16620517,
    9.16277249,
    9.119128,
    9.06260294,
    9.01015779,
    8.95942704,
    8.91830151,
    8.87442665,
    8.83969915,
    8.8362406,
    8.79850223,]

maturity_usd = [3,6,9,12,15,18,21,24,27,30,33,36,39,48,60,72,84,96,108,120,132,144,180,240,300,360]
maturity_usd= [i * weeks_in_month for i in maturity_usd]
ds_rate_usd = [
    2.84253062,
    2.80778295,
    2.7947691,
    2.78672186,
    2.78148933,
    2.76474858,
    2.74385753,
    2.72299332,
    2.70750966,
    2.69278142,
    2.68063008,
    2.67150303,
    2.66686195,
    2.66005416,
    2.67045368,
    2.68499512,
    2.70525985,
    2.72994097,
    2.75619769,
    2.78172399,
    2.804673,
    2.82275184,
    2.857914,
    2.87560338,
    2.86786144,
    2.85316361]





def get_rates():

    interpolate_rub = CubicSpline(maturity_rub, ds_rate_rub)
    interpolate_usd = CubicSpline(maturity_usd, ds_rate_usd)

    interval_rub = np.arange(0, 54, 2)
    interval_usd = np.arange(0, 54, 2)

    df_rub_int = pd.DataFrame(data={'maturity_rub_2weeks': interval_rub, 'rub_act': interpolate_rub(interval_rub)})
    df_usd_int = pd.DataFrame(data={'maturity_usd_2weeks': interval_usd, 'usd_act': interpolate_usd(interval_usd)})


    df_rub_usd_int=pd.concat([df_usd_int, df_rub_int], axis=1, sort=False)
    df_rub_usd_int=df_rub_usd_int.drop(['maturity_rub_2weeks'], axis=1)
    df_rub_usd_int=df_rub_usd_int.rename(columns={"maturity_usd_2weeks":"maturity"})
    df_rub_usd_int['maturity_frac'] = df_rub_usd_int['maturity']/54

    s=0.0134

    new_rates = pd.concat(
        [
            df_rub_usd_int,
            df_rub_usd_int.diff(1).rename(columns={x:x.replace('act','diff') for x in df_rub_usd_int.columns})
        ],
        axis=1)

    new_rates.fillna(0, inplace=True)
    new_rates['fx_act']=(s*(1+new_rates['usd_diff'])/(1+new_rates['rub_diff']))
    new_rates['fx_diff'] =new_rates['fx_act'].diff()


    curve_rub_act = new_rates.loc[1:,'rub_act']
    curve_usd_act = new_rates.loc[1:,'usd_act']
    curve_fx_act = new_rates.loc[1:,'fx_act']

    curve_rub_diff = new_rates.loc[1:,'rub_diff']
    curve_usd_diff = new_rates.loc[1:,'usd_diff']
    curve_fx_diff = new_rates.loc[1:,'fx_diff']
    init = new_rates.loc[0,['rub_act','usd_act','fx_act']]


    return  (
        curve_rub_act,
        curve_usd_act,
        curve_fx_act,
        curve_rub_diff,
        curve_usd_diff,
        curve_fx_diff,
        init)






def get_decomp():

    fx_raw = pd.read_html('./fx_rates.html')[0]
    fx_raw.columns = [', '.join(x[1:]) for x in fx_raw.columns]
    fx=fx_raw.iloc[:,[0,3]]
    fx.columns = ['date','fx_rate']
    fx.loc[:,'date']=pd.to_datetime(fx['date'], format="%d/%m/%Y")
    fx=fx.sort_values(by='date').reset_index(drop=True)

    fx['fx_rate'] = 1/fx['fx_rate']#считаем рубли в долларах

    rub_irs_raw = pd.read_excel('ROISfix history.xlsx')
    rub_irs=rub_irs_raw.loc[:,['Дата ставки','3M']]
    rub_irs.columns=['date','rub_rate']
    rub_irs.loc[:,'date'] = pd.to_datetime(rub_irs['date'], format="%d-%m-%Y")
    rub_irs = rub_irs.sort_values(by='date').reset_index(drop=True)

    usd_irs_raw = pd.read_excel('./LIBOR USD history.xlsx')
    usd_irs=usd_irs_raw.loc[:,['Date','3M']]
    usd_irs.columns=['date','usd_rate']
    usd_irs.loc[:,'date'] = pd.to_datetime(usd_irs['date'], format="%d.%m.%Y")
    usd_irs = usd_irs.sort_values(by='date').reset_index(drop=True)

    mrg = pd.merge(
        usd_irs,
        pd.merge(
            rub_irs,
            fx,
            left_on='date',
            right_on='date',
            how='outer'

        ),
        left_on='date',
        right_on='date',
        how='outer'
    )

    print(mrg.columns)
    merged_interpolated_diff_corr = mrg.interpolate(method='linear').iloc[3:,1:].diff().iloc[1:].corr()
    decomposed = linalg.cholesky(merged_interpolated_diff_corr)
    return decomposed




