from django.shortcuts import render
from django.http import HttpResponse, FileResponse
from django.template import loader
from .models import data
import pandas as pd
import datetime as dt
from matplotlib import pylab
from pylab import *
import matplotlib.pyplot as plt
from io import BytesIO
import PIL.Image
import PIL
from django.core.files.uploadedfile import InMemoryUploadedFile
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as plticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import keras
import keras.layers as KL
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
import io
import urllib
import base64

matplotlib.use('Agg')


# Create your views here.
def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())

def year(request: HttpResponse):
    year = request.POST['year']
    qs = pd.DataFrame(list(data.objects.values()))
    qs['obstime'] = qs['obstime'].dt.tz_localize(None)

    qs['obstime'] = pd.to_datetime(qs['obstime'])
    qs = qs.set_index(['obstime'])
    qs = qs.loc[year]
    # #resampling by day to get the dates where the blizzard happened
    dates = qs.resample('D').sum()
    dates = dates.drop(['tempr', 'ap', 'ws', 'wd', 'rh'], axis= 1)
    # creating a separate column for the day
    dates['day'] = dates.index.map(lambda x: str(x).split()[0].split('-')[2])
    #taking all the dates when the blizzard occured in a month in tab2
    tab2 = ['No Blizzard'] * 12
    for i in range(12):
        dates2 = (dates[dates.index.month == i+1])
        dates2 = dates2.reset_index()
        for x, y in dates2.iterrows():
            if y['blizzard'] != 0:
                tab2[i] = str(tab2[i]) + ',' + str(y['day'])
    # creating a new dataframe that will be displayed
    new = {'Month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                     'November', 'December']
           }

    df = pd.DataFrame(new, columns=['Month', 'Dates','hours'])
    df['Dates'] = tab2
    df['newdates'] = df['Dates'].str.split('No Blizzard,').str[1]
    # to get the no of hours of blizzards in each month
    # resampling by addition
    monthly = qs.resample('M').sum()
    #removing the extra columns
    monthly = monthly.drop(['tempr', 'ap', 'ws', 'wd', 'rh'], axis= 1)
    # adding proper month name to monthly
    monthly['m'] = monthly.index.map(lambda x: str(x).split()[0].split('-')[1])
    import calendar
    monthly['mname'] = monthly['m'].apply(lambda x: calendar.month_abbr[int(x)])
    monthly = monthly.reset_index()
    g =[0]*12
    for i in range(12):
        for x, v in monthly.iterrows():
            if i+1 == int(v['m']):
                g[i] = v['blizzard']
    df['hours'] = g


    template = loader.get_template('month.html')
    #resending the year to the template to be used in the next view
    context = {
               'year': year,
               'df': df,
               }
    return HttpResponse(template.render(context, request))

def month(request: HttpResponse):

    month = request.POST['mon']
    year = request.POST['year']
    f1 = request.POST['feature1']
    f2 = request.POST['feature2']
    type1 = request.POST['gtype1']
    type2 = request.POST['gtype2']


    qs = pd.DataFrame(list(data.objects.values()))
    qs['obstime'] = qs['obstime'].dt.tz_localize(None)
    #
    qs['obstime'] = pd.to_datetime(qs['obstime'])
    qs = qs.set_index(['obstime'])
    qs = (qs[qs.index.month == int(month)])
    ds = qs[year]
    #finding the no of rows in the dataset
    index = ds.index
    r = len(index)
    ds = ds.reset_index()
    #creating a column with the no of rows same as the dataset
    c = [0]*r
    #filling after blizzard after the range of blizzards
    count = 0
    for x, v in ds.iterrows():
        if v["blizzard"] == 1:
            count = 1
            c[x] = ("blizzard")
        elif v["blizzard"] != 1 and count != 0:
            x = x+6
            for i in range(4):
               c[x-6+i] = ("After Blizzard")
                # print(x)
            count = 0
    #filling before blizzard after the range of blizzards
    count = 0
    for m, row in ds.iloc[::-1].iterrows():
        if row["blizzard"] == 1:
            count = 1
            c[m] = ("blizzard")
        elif row["blizzard"] != 1 and count != 0:
            m = m-6
            for i in range(4):
               c[m+3+i] = ("Before Blizzard")
            count = 0
    # ds2 is created to be able to get the differential graph
    ds2 = ds.diff(axis=0, periods=1)
    ds2["filter"] = c
    ds2['obstime'] = ds['obstime']
    ds2 = ds2.iloc[1:]
    #making a column in ds called filter attatching c to it
    ds["filter"] = c

#plotting1

    fig, ax = plt.subplots(figsize=(40, 20))
    color_dict = {0: 'green', 'Before Blizzard': 'orange', 'blizzard': 'red', 'After Blizzard': 'blue'}

    if type1 == '1':
        ax.scatter(ds['obstime'], ds[f1], s=40, c=[color_dict[i] for i in ds['filter']], alpha=1)



    else:

        ax.plot(ds2['obstime'], ds2[f1], c='green', linewidth= 3)
        color_dict = {0:'green','Before Blizzard': 'orange', 'blizzard': 'red', 'After Blizzard': 'blue'}
        ax.scatter(ds2['obstime'], ds2[f1], s=30, c=[color_dict[i] for i in ds2['filter']], alpha = 1)


    if type2 == '1':
        color_dict = {0: 'y', 'Before Blizzard': 'orange', 'blizzard': 'red', 'After Blizzard': 'blue'}
        ax.scatter(ds['obstime'], ds[f2], s=40, c=[color_dict[i] for i in ds['filter']], alpha=1)


    else:
        ax.plot(ds2['obstime'], ds2[f2], c='y', linewidth= 3)
        color_dict = {0:'y','Before Blizzard': 'orange', 'blizzard': 'red', 'After Blizzard': 'blue'}
        ax.scatter(ds2['obstime'], ds2[f2], s=30, c=[color_dict[i] for i in ds2['filter']], alpha = 1)

    degree_sign = u'\N{DEGREE SIGN}'
    labels = {
        'tempr': 'Temperature (' + degree_sign + 'C)',
        'ap': 'Atmospheric Pressure (mbar)',
        'ws': 'Wind Speed (knots)',
        'rh': 'Relative Humidity (%)',
        'wd': 'Wind direction(Deg)',

    }
    legend = {
        'tempr': 'Temperature',
        'ap': 'Atmospheric Pressure',
        'ws': 'Wind Speed',
        'rh': 'Relative Humidity',
        'wd': 'Wind direction(Deg)',
    }
    kind ={
        '1': ' vs time',
        '2': ' Differential'

    }

    orange_patch = mpatches.Patch(color='orange', label='4 Hrs Before Blizzard')
    red_patch = mpatches.Patch(color='red', label='Blizzard')
    blue_patch = mpatches.Patch(color='blue', label='4 Hrs After Blizzard')
    gold_patch = mpatches.Patch(color='y', label=legend[f2] + kind[type2])
    green_patch = mpatches.Patch(color='green', label=legend[f1] + kind[type1])
    # grey_patch = mpatches.Patch(color='green', label='')
    if (f1 == f2 and type1 == type2):
        ax.legend(handles=[red_patch, orange_patch, blue_patch, gold_patch], fontsize=30)
        ax.set_ylabel(labels[f1] , fontsize=34)
    else:
        ax.legend(handles=[red_patch, orange_patch, blue_patch, gold_patch, green_patch], fontsize=30)
        ax.set_ylabel(labels[f1] + ' and ' + labels[f2], fontsize=34)

    plt.grid()
    fig.text(0.6, 0.97, "Blizzard Prediction / Pattern System", fontsize=35, ha='right')
    fig.text(0.7, 0.94, "Digital Current Weather Information System (DCWIS), Bharati Station, Antarctica ", fontsize=30, ha='right')

    ax.set_title('Blizzards during '+'0'+ str(month)+'-' + str(year), fontsize=50)
    ax.set_xlabel('Time', fontsize=34)


    xticks = ax.xaxis.get_major_ticks()
    yticks = ax.yaxis.get_major_ticks()
    for i in range(len(xticks)):
        xticks[i].label.set_fontsize(20)
        xticks[i].label.set_rotation('vertical')
    for i in range(len(yticks)):
        yticks[i].label.set_fontsize(20)

    loc = plticker.MultipleLocator(base=1)
    ax.xaxis.set_major_locator(loc)

    # Store image in a bytes buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()

    # Send buffer in a http response the the browser with the mime type image/png set
    return HttpResponse(buffer.getvalue(), content_type="image/png")
    # return HttpResponse(ds2_b.to_html())
def pred(request: HttpResponse):

    kind = request.POST['prediction']
    feature = request.POST['feature']
    start_date = request.POST['start']
    end_date = request.POST['end']
    df = pd.DataFrame(list(data.objects.values()))



    if kind == 'rnn':
        df.index = df['obstime']
        df.index = pd.to_datetime(df.index)
        df.dropna(subset=["blizzard"], inplace=True)
        del df['obstime']
        df_features = df.iloc[:, 0:5]


        df_target = df.iloc[:, 5:]
        X_train = df_features.iloc[0:-3980, :]
        X_test = df_features.iloc[-3980:, :]
        y_train = df_target.iloc[0:-3980].values
        y_test = df_target.iloc[-3980:].values
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        rtime = X_test.index
        # from sklearn.preprocessing import MinMaxScaler
        scalar = MinMaxScaler()


        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        inputs = KL.Input(shape=(1, 5))
        x = KL.LSTM(units=8, activation='relu')(inputs)
        outputs = KL.Dense(units=1, activation='sigmoid')(x)
        import keras

        model = keras.models.Model(inputs, outputs)
        opt = keras.optimizers.Adamax()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=16, epochs=15, class_weight={0: 1, 1: 2})
        y_pred = model.predict(X_test)
        print(y_pred)

        y_pred = [1 if y >= 0.45 else 0 for y in y_pred]
        print(np.sum(y_pred))

        # print(classification_report(y_test, y_pred))
        X_train = df_features.iloc[0:-3980, :]
        X_test = df_features.iloc[-3980:, :]
        y_train = df_target.iloc[0:-3980].values
        y_test = df_target.iloc[-3980:].values
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        dfy = pd.DataFrame(data={'ws': X_test.ws, 'ap': X_test.ap, 'rh': X_test.rh, 'tempr': X_test.tempr, 'wd': X_test.wd,
                                 'actual': y_test.flatten(), 'pred': y_pred})

        dfy = dfy.sort_values('obstime')
        cr = classification_report(y_test, y_pred,output_dict=True)
        cr = pd.DataFrame(cr).transpose()
        cr.rename(columns={'f1-score': 'f1'}, inplace=True)

    if kind == 'LSTM':
        df.index = df['obstime']
        df.index = pd.to_datetime(df.index)
        df.dropna(subset=["blizzard"], inplace=True)
        del df['obstime']
        df_features = df.iloc[:, 0:5]


        df_target = df.iloc[:, 5:]
        X_train = df_features.iloc[0:-3980, :]
        X_test = df_features.iloc[-3980:, :]
        y_train = df_target.iloc[0:-3980].values
        y_test = df_target.iloc[-3980:].values
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        rtime = X_test.index
        # from sklearn.preprocessing import MinMaxScaler
        scalar = MinMaxScaler()


        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        from tensorflow import keras
        model = keras.Sequential()
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=64,
                    input_shape=(X_train.shape[1], X_train.shape[2])
                )
            )
        )
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(units=1))
        opt = keras.optimizers.Adamax()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=32, epochs=15, class_weight={0: 1, 1: 2})
        y_pred = model.predict(X_test)


        y_pred = [1 if y >= 0.55 else 0 for y in y_pred]
        print(np.sum(y_pred))

        # print(classification_report(y_test, y_pred))
        X_train = df_features.iloc[0:-3980, :]
        X_test = df_features.iloc[-3980:, :]
        y_train = df_target.iloc[0:-3980].values
        y_test = df_target.iloc[-3980:].values
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        dfy = pd.DataFrame(data={'ws': X_test.ws, 'ap': X_test.ap, 'rh': X_test.rh, 'tempr': X_test.tempr, 'wd': X_test.wd,
                                 'actual': y_test.flatten(), 'pred': y_pred})

        dfy = dfy.sort_values('obstime')
        cr = classification_report(y_test, y_pred,output_dict=True)
        cr = pd.DataFrame(cr).transpose()
        cr.rename(columns={'f1-score': 'f1'}, inplace=True)


    if kind == 'ann':
        df.index = df['obstime']
        df.index = pd.to_datetime(df.index)
        df.dropna(subset=["blizzard"], inplace=True)
        del df['obstime']
        df_features = df.iloc[:, 0:5]


        df_target = df.iloc[:, 5:]
        X_train = df_features.iloc[0:-3980, :]
        X_test = df_features.iloc[-3980:, :]
        y_train = df_target.iloc[0:-3980].values
        y_test = df_target.iloc[-3980:].values
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        rtime = X_test.index
        # from sklearn.preprocessing import MinMaxScaler
        scalar = MinMaxScaler()


        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)
        model = Sequential()
        model.add(KL.Dense(20, input_dim=5, activation='relu'))
        model.add(KL.Dense(14, activation='relu'))
        model.add(KL.Dense(8, activation='relu'))
        model.add(KL.Dense(1, activation='sigmoid'))
        import keras
        opt = keras.optimizers.Adamax()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=8, epochs=12, class_weight={0: 1, 1: 2})
        y_pred = model.predict(X_test)
        print(y_pred)

        y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
        print(np.sum(y_pred))

        # print(classification_report(y_test, y_pred))
        X_train = df_features.iloc[0:-3980, :]
        X_test = df_features.iloc[-3980:, :]
        y_train = df_target.iloc[0:-3980].values
        y_test = df_target.iloc[-3980:].values
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        dfy = pd.DataFrame(data={'ws': X_test.ws, 'ap': X_test.ap, 'rh': X_test.rh, 'tempr': X_test.tempr, 'wd': X_test.wd,
                                 'actual': y_test.flatten(), 'pred': y_pred})

        dfy = dfy.sort_values('obstime')
        cr = classification_report(y_test, y_pred,output_dict=True)
        cr = pd.DataFrame(cr).transpose()
        cr.rename(columns={'f1-score': 'f1'}, inplace=True)

    if kind == 'cnn':
        df.index = df['obstime']
        df.index = pd.to_datetime(df.index)
        df.dropna(subset=["blizzard"], inplace=True)
        del df['obstime']
        df = df.reset_index(level=None)

        dfa = df.to_numpy()

        # split a multivariate sequence into samples
        def split_sequences(sequences, n_steps):
            X, y = list(), list()
            for i in range(len(sequences)):
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the dataset
                if end_ix > len(sequences):
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y)

        n_steps = 7
        X, y = split_sequences(dfa[:, 0:], n_steps)
        X_train = X[0:-3980, :, :]
        X_test = X[-3980:, :, :]
        y_train = y[0:-3980]
        y_test = y[-3980:]
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        rtime = X_test[:, :, 0]

        X_test = X_test[:, :, 1:]
        X_train = X_train[:, :, 1:]
        X_test = X_test.astype(np.float32)
        X_train = X_train.astype(np.float32)
        n_features=5
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=50, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=16, epochs=30)
        y_pred = model.predict(X_test, verbose=0)
        y_pred = [1 if y >= 0.455 else 0 for y in y_pred]
        dfy = pd.DataFrame(
            data={'obstime': rtime[:, 0], 'tempr': X_test[:, 0, 0].flatten(), 'ap': X_test[:, 0, 1].flatten(),
                  'ws': X_test[:, 0, 2].flatten(), 'wd': X_test[:, 0, 3].flatten(), 'rh': X_test[:, 0, 4].flatten(),
                  'actual': y_test, 'pred': y_pred})
        dfy = dfy.sort_values('obstime')
        dfy = dfy.set_index('obstime')
        cr = classification_report(y_test, y_pred,output_dict=True)
        cr = pd.DataFrame(cr).transpose()
        cr.rename(columns={'f1-score': 'f1'}, inplace=True)


    dfy['result'] = 'no'
    for i in range(len(dfy)):
        if (dfy['pred'][i] == 1):
            if (dfy['actual'][i] == 1):
                dfy['result'][i] = 'correct'
            if (dfy['actual'][i] == 0):
                dfy['result'][i] = 'incorrect'
        else:
            dfy['result'][i] = 'no'
    dfy = dfy[start_date: end_date]

    degree_sign = u'\N{DEGREE SIGN}'
    labels = {
        'tempr': 'Temperature (' + degree_sign + 'C)',
        'ap': 'Atmospheric Pressure (mbar)',
        'ws': 'Wind Speed (knots)',
        'rh': 'Relative Humidity (%)'
    }
    fig = plt.figure(figsize=(20, 10))
    fig, ax1 = plt.subplots(figsize=(100, 60))

    ax1.plot(dfy.index, dfy[feature], color='#3366ff', alpha=1)
    ax1.scatter(dfy[dfy['actual'] == 1].index, dfy[dfy['actual'] == 1][feature], color='#F4B400', s=5500, label='Actual')
    ax1.scatter(dfy[dfy['result'] == 'correct'].index, dfy[dfy['result'] == 'correct'][feature], color='green', s=1500,
                label='Correct')
    ax1.scatter(dfy[dfy['result'] == 'incorrect'].index, dfy[dfy['result'] == 'incorrect'][feature], color='red', s=1000,
                label='Incorrect')

    ax1.set_ylim(0, 60)

    for i in range(0, 60, 5):
        plt.axhline(y=i, c='black')

    fig.legend(loc="upper right", fontsize=100)
    ax1.tick_params(labelsize=80)
    ax1.set_title('Predictions(blizzard occurence) wrt ' +labels[feature], fontsize=100)
    ax1.set_xlabel('Time', fontsize=100)
    ax1.set_ylabel(labels[feature], fontsize=100)
    # Store image in a bytes buffer
    buffer = BytesIO()
    canvas = pylab.get_current_fig_manager().canvas
    canvas.draw()
    pilImage = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    pilImage.save(buffer, "PNG")
    pylab.close()

    template = loader.get_template('prediction.html')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    context = {
        'pred': uri,
        'kind': kind,
        'cr' : cr,

    }
    return HttpResponse(template.render(context, request))


# def cf(request: HttpResponse):
#
#     kind = request.POST['kind']
#     mon = request.POST['mon']
#     cr = request.POST['cr']
#     pred = '../../static/'
#     pred += kind + '/conf/' + mon + '.jpeg'
#     cr = pd.DataFrame(cr).transpose()
#     return HttpResponse(cr.to_html())




