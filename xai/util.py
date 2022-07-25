import numpy as np
import pandas as pd
import sklearn


def extract_cat_col_mappings():
    """
    Get explanation column names to match columns in original data
    Used in xai_to_categorical()
    """

    def drop_assignments(in_str):
        """Helper fn for removing the _entry appendix for categorical column names"""
        if in_str in ['Betrag_5', 'Kreditkontr_betrag']:  # not categorical, omit
            return in_str
        split = str(in_str).split('_')
        return '_'.join(in_str.split('_')[:-1]) if len(split) > 1 else in_str

    # load data & explanation
    train = pd.read_csv('./data/normal_2.csv', encoding='ISO-8859-1')
    shap_explanation = pd.read_csv('./outputs/shap_eval.csv', index_col=0)
    # X_train, X_eval, X_test, y_eval, y_test = load_splits(source_folder='./data', mode='prep', keep_anomalies=True)

    # remove _assignments from categorical columns
    drop_assignments = np.vectorize(drop_assignments)
    cols = drop_assignments(shap_explanation.columns.values)

    # find unique column names and check if they are the same cols that are in original data
    unique_cols = np.unique(cols, return_counts=True)
    col_dict = {key: val for key, val in zip(unique_cols[0], unique_cols[1])}
    assert not (np.sort(unique_cols[0]) != np.sort(
        train.drop(["Label", "Belegnummer", "Position", "Transaktionsart", "Erfassungsuhrzeit"],
                   axis=1).columns.values)).any()

    # Get list of which columns belong together
    i = 0
    col_names = []
    col_mapping = []
    for col in cols:
        if col in col_dict:
            col_names.append(col)
            col_mapping.append(list(range(i, i + col_dict[col])))
            i += col_dict[col]
            col_dict.pop(col)
    # check that all mappings are correct
    for col_list in col_mapping:
        assert np.unique([cols[x] for x in col_list]).shape[0] == 1

    return col_names, col_mapping


def scores_to_categorical(data, categories):
    """np.concatenate(data_cat, data[:, 29:])
    Slims a data array by adding column values of rows together for all column pairs in list categories.
    Used for summing up scores that were calculated for one-hot representation of categorical features.
    Gives a score for each categorical feature.
    :param data:        np.array of shape (samples, features) with scores from one-hot features
    :param categories:  list with number of features that were used for one-hot encoding each categorical feature
                        (as given by sklearn.OneHotEncoder.categories_)
    :return:
    """
    import numpy as np
    data_cat = np.zeros((data.shape[0], len(categories)))
    for i, cat in enumerate(categories):
        data_cat[:, i] = np.sum(data[:, cat], axis=1)
    if data.shape[1] > len(categories):  # get all data columns not in categories and append data_cat
        categories_flat = [item for sublist in categories for item in sublist]
        data_cat = np.concatenate((data[:, list(set(range(data.shape[1])) ^ set(categories_flat))], data_cat), axis=1)
    return data_cat


def xai_to_categorical(expl_df, dataset_name, language='en'):
    """Converts XAI scores to categorical values and adds column names
    Example:
    xai_to_categorical(xai_score_path='./scoring/outputs/ERPSim_BSEG_RSEG/pos_shap.csv',
                       out_path='./scoring/outputs/ERPSim_BSEG_RSEG/joint_shap.csv',
                       data_path='./datasets/real/ERPSim/BSEG_RSEG/ERP_Fraud_PJS1920_BSEG_RSEG.csv')
    """
    index = expl_df.index
    if dataset_name == 'ex1':
        cat_cols = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21],
                    [22, 23], [24, 25], [26, 27], [28, 29], [30], [31, 32], [33, 34], [35, 36], [37, 38], [39, 40],
                    [41, 42], [43, 44], [45, 46], [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
                    [63, 64, 65], [66, 67, 68], [69, 70], [71, 72, 73],
                    [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86], [87, 88, 89, 90], [91, 92, 93],
                    [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                     115, 116, 117, 118, 119, 120, 121, 122],
                    [123, 124, 125, 126],
                    [127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
                    [143, 144, 145], [146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157],
                    [158, 159, 160], [161, 162, 163], [164, 165, 166, 167, 168, 169],
                    [170, 171, 172, 173, 174, 175, 176], [177, 178, 179, 180, 181, 182], [183, 184, 185, 186, 187, 188],
                    [189, 190, 191, 192, 193, 194], [195, 196, 197, 198, 199, 200], [201, 202, 203, 204, 205, 206],
                    [207, 208, 209, 210, 211, 212], [213, 214, 215, 216, 217, 218], [219, 220, 221, 222, 223, 224],
                    [225, 226, 227, 228, 229, 230], [231, 232, 233, 234, 235, 236], [237, 238, 239, 240, 241, 242]]
    elif dataset_name == 'ex2':
        cat_cols = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21],
                    [22, 23], [24, 25], [26, 27], [28, 29], [30], [31, 32], [33, 34], [35, 36], [37, 38], [39, 40],
                    [41, 42], [43, 44], [45, 46], [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
                    [63, 64, 65], [66, 67, 68], [69, 70], [71, 72, 73],
                    [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85], [86, 87, 88, 89], [90, 91, 92],
                    [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
                     114, 115, 116], [117, 118, 119, 120], [121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131],
                    [132, 133, 134],
                    [135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
                     155, 156, 157], [158, 159, 160], [161, 162, 163], [164, 165, 166, 167, 168],
                    [169, 170, 171, 172, 173, 174], [175, 176, 177, 178, 179], [180, 181, 182, 183, 184, 185],
                    [186, 187, 188, 189, 190, 191], [192, 193, 194, 195, 196, 197], [198, 199, 200, 201, 202, 203],
                    [204, 205, 206, 207, 208, 209], [210, 211, 212, 213, 214, 215], [216, 217, 218, 219, 220, 221],
                    [222, 223, 224, 225, 226, 227], [228, 229, 230, 231, 232, 233], [234, 235, 236, 237, 238, 239]]

    col_names_de = ['Bestandskonto', 'Bewertungsklasse',
                    'Einzelpostenanzeige moeglich', 'Erfolgskontentyp', 'Geschaeftsbereich', 'Gruppenkennzeichen',
                    'KZ EKBE', 'Kennzeichen: Posten nicht kopierbar ?', 'Kostenstelle', 'Kreditkontr_Bereich',
                    'Laufende Kontierung', 'PartnerPrctr', 'Profitcenter', 'Rechnungsbezug', 'Soll/Haben-Kennz_',
                    'Sperrgrund Menge', 'Steuerkennzeichen', 'Bewertungskreis', 'Steuerstandort', 'Umsatzwirksam',
                    'Verwaltung offener Posten', 'Werk', 'Zahlungsbedingung', 'Zahlungssperre',
                    'Alternative Kontonummer',
                    'Basismengeneinheit', 'BestPreisMngEinheit', 'Bestandsbuchung', 'Bestellmengeneinheit',
                    'Buchungsschluessel', 'Buchungszeilen-Id', 'ErfassungsMngEinh', 'Hauptbuchkonto', 'Kontoart',
                    'Kostenart', 'Kreditor', 'Material', 'Preissteuerung', 'Sachkonto', 'Vorgang', 'Vorgangsart GL',
                    'Wertestring', 'Betrag Hauswaehr', 'Betrag', 'Betrag_5', 'Gesamtbestand', 'Gesamtwert',
                    'Kreditkontr_betrag', 'Menge in BPME', 'Menge in ErfassME', 'Menge', 'Skontobasis']

    col_names_en = ['inventory account', 'valuation class', 'line item display possible', 'P&L statement account type',
                    'business unit', 'group indicator', 'IND EKBE', 'IND items not copyable', 'cost centre',
                    'credit control area', 'regular account assignment', 'partner profit center', 'profit center',
                    'invoice reference', 'debit/credit indicator', 'blocking reason quantity', 'tax code',
                    'valuation area', 'jurisdiction', 'sales-related', 'open item management', 'plant',
                    'terms of payment', 'payment block', 'alternative account number', 'base unit of measure',
                    'purchase order price unit', 'stock posting', 'order unit', 'posting key', 'posting line id',
                    'unit of entry', 'G/L account', 'account type', 'cost element', 'vendor', 'material',
                    'price control', 'general ledger account', 'transaction', 'G/L transaction type', 'value string',
                    'amount in local currency', 'amount', 'amount_5', 'total stock', 'total value',
                    'credit control amount', 'quantity in order unit', 'quantity in entry unit',
                    'quantity', 'cash discount base']

    col_names = col_names_de if language == 'ger' else col_names_en
    expls_joint = scores_to_categorical(expl_df.values, cat_cols)

    return pd.DataFrame(expls_joint, index=index, columns=col_names)
