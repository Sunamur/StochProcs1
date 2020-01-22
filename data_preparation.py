import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
from scipy.interpolate import CubicSpline





def get_rates():

    maturity_rub = [3,6,9,12,24,36,48,60,72,84,96,108,120,144,180,]
    maturity_rub= [i * 4 for i in maturity_rub]


    ds_rate_rub = [8.51987757,8.68611319,8.90460383,9.04938647,9.16620517,9.16277249,9.119128,9.06260294,9.01015779,
                8.95942704,8.91830151,8.87442665,8.83969915,8.8362406,8.79850223,]

    interpolate_rub = CubicSpline(maturity_rub, ds_rate_rub)
    interval_rub = np.arange(0, 54, 2)

    maturity_usd = [3,6,9,12,15,18,21,24,27,30,33,36,39,48,60,72,84,96,108,120,132,144,180,240,300,360]
    maturity_usd= [i * 4 for i in maturity_usd]
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
        2.68063008,2.67150303,2.66686195,2.66005416,2.67045368,2.68499512,2.70525985,2.72994097,2.75619769,2.78172399,
                2.804673,2.82275184,2.857914,2.87560338,2.86786144,2.85316361]


    interpolate_usd = CubicSpline(maturity_usd, ds_rate_usd)
    interval_usd = np.arange(0, 54, 2)

    d = {'maturity_rub': maturity_rub, 'ds_rate_rub': ds_rate_rub}
    d1={'maturity_usd': maturity_usd, 'ds_rate_usd': ds_rate_usd}
    df_rub = pd.DataFrame(data=d)
    df_usd = pd.DataFrame(data=d1)

    d2={'maturity_rub_2weeks': interval_rub, 'ds_rate_int_rub': interpolate_rub(interval_rub)}
    d3={'maturity_usd_2weeks': interval_usd, 'ds_rate_int_usd': interpolate_usd(interval_usd)}
    df_rub_int = pd.DataFrame(data=d2)
    df_usd_int = pd.DataFrame(data=d3)


    df_rub_usd_int=pd.concat([df_usd_int, df_rub_int], axis=1, sort=False)
    df_rub_usd_int=df_rub_usd_int.drop(['maturity_rub_2weeks'], axis=1)
    df_rub_usd_int=df_rub_usd_int.rename(columns={"maturity_usd_2weeks":"maturity"})
    s=0.0134
    df_rub_usd_int['FX']=s*(1+df_rub_usd_int['ds_rate_int_usd'])/(1+df_rub_usd_int['ds_rate_int_rub'])

    ds_rate_rub_diff=np.diff(interpolate_rub(interval_rub))
    ds_rate_usd_diff=np.diff(interpolate_usd(interval_usd))
    fx_diff=np.diff(np.log(s*(1+interpolate_usd(interval_usd))/(1+interpolate_rub(interval_rub))))
    maturity_diff=np.diff(interval_rub)


    df_diff = pd.DataFrame(data={'maturity_diff':maturity_diff,'ds_rate_rub_diff':ds_rate_rub_diff, 
                                'ds_rate_usd_diff':ds_rate_usd_diff, 'fx_diff':fx_diff })

    curve_rub = df_diff.loc[:,'ds_rate_rub_diff']
    curve_us = df_diff.loc[:,'ds_rate_usd_diff']
    curve_fx = df_diff.loc[:,'fx_diff']

    return  curve_us, curve_rub, curve_fx, df_rub_usd_int.iloc[0,1:].values


def get_actual_rates():
    curve_usd, curve_rub, curve_fx, init = get_rates()
    curve_usd_act = curve_usd.cumsum()+init[0]
    curve_rub_act = curve_rub.cumsum()+init[1]
    curve_fx_act  = curve_fx.cumsum() +init[2]
    return curve_usd_act, curve_rub_act, curve_fx_act, init




def get_decomp():

    fx_raw = pd.read_html('./fx_rates.html')[0]
    fx_raw.columns = [', '.join(x[1:]) for x in fx_raw.columns]
    fx=fx_raw.iloc[:,[0,3]]
    fx.columns = ['date','fx_rate']
    fx.loc[:,'date']=pd.to_datetime(fx['date'], format="%d/%m/%Y")
    fx=fx.sort_values(by='date').reset_index(drop=True)

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




