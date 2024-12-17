# def correlation_map(Y, effect_types, model=None, Y2X=None, method='pearson'):
#     assert model is not None or Y2X is not None, 'Must specify either the model or Y2X'

#     # Parse effect types
#     if isinstance(effect_types, dict):
#         # Detect effect types
#         col_names = list(effect_types.keys())
#         effect_types = np.array(list(effect_types.values()))
#     else:
#         # No column names known
#         col_names = None
#     col_names_model = None

#     # Convert Y to numpy
#     if isinstance(Y, pd.DataFrame):
#         if col_names is not None:
#             Y = Y[col_names]
#         else:
#             col_names = list(Y.columns)
#         Y = Y.to_numpy()

#     # Encode Y
#     Y = encode_design(Y, effect_types)
    
#     # Create Y2X function
#     if Y2X is None:
#         # Convert model to numpy
#         if isinstance(model, pd.DataFrame):
#             if col_names is not None:
#                 model = model[col_names]
#             model = model.to_numpy()

#         # Encode the model
#         modelenc = encode_model(model, effect_types)

#         # Create Y2X
#         Y2X = numba.njit(lambda Y: x2fx(Y, modelenc))

#         # Create the columns names
#         cn = col_names if col_names is not None else list(np.arange(len(effect_types)).astype(str))
#         cn_enc = encode_names(cn, effect_types)
#         col_names_model = model2names(modelenc, cn_enc)

#     # Create X
#     X = Y2X(Y)

#     # Compute the correlations
#     corr = pd.DataFrame(X, columns=col_names_model).corr(method=method)
#     if col_names_model is not None:
#         return corr
#     else:
#         return corr.to_numpy()