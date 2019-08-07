Definition of 7 types of models overall: 
- DNN
- Ensemble Stacking Classifier and
- SklearnExtendedClassifier (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, SVC, XGBoost)

A `GenericClassifier` class has been defined within the `generic_classifier` module which is inherited by both the `DnnClassifier` and `SklearnExtendedClassifier` classes. 
This pattern offers abstraction of some methods (for training, evaluation, extraction/plotting of ROC curves, post-processing of results, etc.) and allows for fitting a model irrespective of the type/architecture of the model itself.
