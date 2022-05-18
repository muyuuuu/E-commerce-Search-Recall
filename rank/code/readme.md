#  
- data.py: for processing data
- rank_model.py: model for training
- trainer.py: for training model 
- wrapper.py: modify the trained model's inputs and output to the format of the competition, and convert a checkpoint to saved model  
- bert: bert model

## training
### data
   - We provide example in directory **data**.
### training scripts
   - use `python trainer.py --arguments xx`

## convert a checkpoint model to saved_model and wrap the model to meet the requirements of the competition
   - use `python wrapper.py --arguments xx`
