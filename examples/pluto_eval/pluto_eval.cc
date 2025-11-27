#include "pluto_eval.h"

int pluto_eval_main()
{
	Eigen::MatrixXd X_test;
	Eigen::MatrixXd y_test_coords;
	NEURAL_NETWORK::Helpers::ReadFromCSVIntoEigen("data/Pluto/rx_tx_test.csv", 
												  X_test, 
												  y_test_coords, 
												  ',');
	Eigen::MatrixXd X_test_unscaled = X_test;
	NEURAL_NETWORK::Helpers::ScaleData(X_test);

	std::set<Symbol> unique_symbols;
	for (int i = 0; i < y_test_coords.rows(); i++)
	{
		unique_symbols.insert({y_test_coords(i, 0), y_test_coords(i, 1)});
	}

	std::map<Symbol, int> symbol_to_label;
	int next_label = 0;
	for (const auto &sym : unique_symbols)
	{
		symbol_to_label[sym] = next_label++;
	}

	Eigen::MatrixXd y_test(y_test_coords.rows(), 1);
	for (int i = 0; i < y_test_coords.rows(); i++)
	{
		Symbol key{y_test_coords(i, 0), y_test_coords(i, 1)};
		y_test(i, 0) = symbol_to_label.count(key) ? symbol_to_label[key] : -1;
	}

	NEURAL_NETWORK::Model model2;
	model2.LoadModel("data/pluto_model_save.bin");

	std::map<int, Symbol> label_to_symbol;
	for (auto const &kv : symbol_to_label)
	{
		label_to_symbol[kv.second] = kv.first;
	}

	std::cout << "\nEvaluating loaded model on test data:\n";
	model2.Evaluate(X_test, y_test, BATCH_SIZE);

	Eigen::MatrixXd y_pred_labels = model2.Predict(X_test, 1);

	std::vector<double> x_input, y_input, x_true, y_true, x_pred, y_pred;

	for (int i = 0; i < X_test_unscaled.rows(); i++)
	{
		x_input.push_back(X_test_unscaled(i, 0));
		y_input.push_back(X_test_unscaled(i, 1));
		x_true.push_back(y_test_coords(i, 0));
		y_true.push_back(y_test_coords(i, 1));
		int predicted_label = static_cast<int>(y_pred_labels(i, 0));
		if (label_to_symbol.count(predicted_label))
		{
			Symbol coords = label_to_symbol[predicted_label];
			x_pred.push_back(coords.first);
			y_pred.push_back(coords.second);
		}
	}

	plt::figure_size(1200, 1000);
	plt::named_plot("Input", x_input, y_input, "o");
	plt::named_plot("True Values", x_true, y_true, "o");
	plt::named_plot("Predictions", x_pred, y_pred, "o");
	plt::title("Symbol Prediction");
	plt::xlabel("I");
	plt::ylabel("Q");
	plt::legend();
	plt::grid(true);
	plt::show();

	return 0;
}