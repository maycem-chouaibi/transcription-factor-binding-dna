import data_handling as dh
import model as m

train_features, test_features, train_labels, test_labels = dh.split_data(*dh.one_hot_encode(*dh.load_preprocess_data()))
model = m.create_model(train_features)
# Train the model
model.fit(train_features, train_labels, epochs=10, batch_size=100)
