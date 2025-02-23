from sklearn.ensemble import \
     (RandomForestClassifier as RF,
      GradientBoostingClassifier as GBR)
from sklearn.metrics import confusion_matrix, classification_report

# # Splitting into 80% training and 20% testing
# # Splitting into features (X) and target variable (y) if applicable
# x = data_lending_clean.iloc[:, :-1]  # All columns except the last as features
# y = data_lending_clean.iloc[:, -1]   # Assuming the last column is the target variable
#
# # Splitting into 80% training and 20% testing
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)
# print("Training set shape:", x_train.shape)
# print("Testing set shape:", x_test.shape)
#
# #Random Forest
# bag_lending = RF(max_features=x_train.shape[1], random_state=0)
# bag_lending.fit(x_train, y_train)
# y_pred = bag_lending.predict(x_test)
# conf_matrix = confusion_matrix(y_test, y_pred)
# # Plot confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()
# print(classification_report(y_test, y_pred))
#
# # Boosting
# boost_lending = GBR(n_estimators=5000,
#                    learning_rate=0.001,
#                    max_depth=3,
#                    random_state=0,
#                    verbose = 1)
# boost_lending.fit(x_train, y_train)
# y_pred_boost = boost_lending.predict(x_test)
# conf_matrix_boost = confusion_matrix(y_test, y_pred_boost)
# # Plot confusion matrix
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix_boost, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()