from kfp import dsl, compiler
from typing import List, Dict
import os
from kfp import kubernetes
from kfp.dsl import Input, Output, Dataset, Model

IMAGE_TAG = '0.0.13'
# IMAGE_TAG = 'latest'

@dsl.component(
    base_image=f"quay.io/ecosystem-appeng/rec-sys-app:{IMAGE_TAG}", packages_to_install=['grpcio'])
def generate_candidates(item_input_model: Input[Model], user_input_model: Input[Model], item_df_input: Input[Dataset], user_df_input: Input[Dataset]):
    from feast import FeatureStore
    from feast.data_source import PushMode
    from models.data_util import data_preproccess
    from models.user_tower import UserTower
    from models.item_tower import ItemTower
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import torch
    import os
    print('WOWOWWO')
    print(os.listdir('.'))
    print(os.listdir('feature_repo/'))
    store = FeatureStore(repo_path="feature_repo/")
    
    item_encoder = ItemTower()
    user_encoder = UserTower()
    item_encoder.load_state_dict(torch.load(item_input_model.path))
    user_encoder.load_state_dict(torch.load(user_input_model.path))
    item_encoder.eval()
    user_encoder.eval()
    # load item and user dataframes
    item_df = pd.read_parquet(item_df_input.path)
    user_df = pd.read_parquet(user_df_input.path)
    
    # Create a new table to be push to the online store
    item_embed_df = item_df[['item_id']].copy()
    user_embed_df = user_df[['user_id']].copy()

    # Encode the items and users
    item_embed_df['embedding'] = item_encoder(**data_preproccess(item_df)).detach().numpy().tolist()
    user_embed_df['embedding'] = user_encoder(**data_preproccess(user_df)).detach().numpy().tolist()

    # Add the currnet timestamp
    item_embed_df['event_timestamp'] = datetime.now()
    user_embed_df['event_timestamp'] = datetime.now()

    # Push the new embedding to the offline and online store
    store.push('item_embed_push_source', item_embed_df, to=PushMode.ONLINE)
    store.push('user_embed_push_source', user_embed_df, to=PushMode.ONLINE)
    
    # Materilize the online store
    store.materialize_incremental(datetime.now(), feature_views=['item_embedding', 'user_items', 'item_features'])

    # Calculate user recommendations for each user
    item_embedding_view = 'item_embedding'
    k = 64
    # # Feast have bug here need to be fixed
    # item_recommendation = []
    # for user_embed in user_embed_df['embedding']:
    #     item_recommendation.append(
    #         store.retrieve_online_documents(
    #             query=user_embed,
    #             top_k=k,
    #             features=[f'{item_embedding_view}:item_id', f'{item_embedding_view}:embeddings']
    #         ).to_df()
    #     )
    item_recommendation = [np.random.randint(0, len(user_embed_df), k).tolist()] *len (user_embed_df)

    # Pushing the calculated items to the online store
    user_items_df = user_embed_df[['user_id']].copy()
    user_items_df['event_timestamp'] = datetime.now()
    user_items_df['top_k_item_ids'] = item_recommendation

    store.push('user_items_push_source', user_items_df, to=PushMode.ONLINE)


@dsl.component(base_image=f"quay.io/ecosystem-appeng/rec-sys-app:{IMAGE_TAG}",)
def train_model(item_df_input: Input[Dataset], user_df_input: Input[Dataset], interaction_df_input: Input[Dataset], neg_interaction_df_input:Input[Dataset], item_output_model: Output[Model], user_output_model: Output[Model]):
    from models.user_tower import UserTower
    from models.item_tower import ItemTower
    from models.train_two_tower import train_two_tower
    import pandas as pd
    import torch
    dim = 64

    item_df = pd.read_parquet(item_df_input.path)
    user_df = pd.read_parquet(user_df_input.path)
    interaction_df = pd.read_parquet(interaction_df_input.path)
    neg_interaction_df = pd.read_parquet(neg_interaction_df_input.path)

    item_encoder = ItemTower(dim)
    user_encoder = UserTower(dim)
    train_two_tower(item_encoder, user_encoder, item_df, user_df, interaction_df, neg_interaction_df)
    
    torch.save(item_encoder.state_dict(), item_output_model.path)
    torch.save(user_encoder.state_dict(), user_output_model.path)
    item_output_model.metadata['framework'] = 'pytorch'
    user_output_model.metadata['framework'] = 'pytorch'
    
@dsl.component(
    base_image=f"quay.io/ecosystem-appeng/rec-sys-app:{IMAGE_TAG}", packages_to_install=['grpcio'])
def load_data_from_feast(item_df_output: Output[Dataset], user_df_output: Output[Dataset], interaction_df_output: Output[Dataset], neg_interaction_df_output: Output[Dataset]):
    from feast import FeatureStore
    from datetime import datetime
    import pandas as pd
    import os
    print('WOWOWWO')
    print(os.listdir('.'))
    print(os.listdir('feature_repo/'))
    store = FeatureStore(repo_path="feature_repo/")
    # load feature services
    item_service = store.get_feature_service("item_service")
    user_service = store.get_feature_service("user_service")
    interaction_service = store.get_feature_service("interaction_service")
    neg_interactions_service = store.get_feature_service('neg_interaction_service')

    num_users = 1_000
    n_items = 5_000

    user_ids = list(range(1, num_users+ 1))
    item_ids = list(range(1, n_items+ 1))

    # select which items to use for the training
    item_entity_df = pd.DataFrame.from_dict(
        {
            'item_id': item_ids,
            'event_timestamp': [datetime(2025, 1, 1)] * len(item_ids) 
        }
    )
    # select which users to use for the training
    user_entity_df = pd.DataFrame.from_dict(
        {
            'user_id': user_ids,
            'event_timestamp': [datetime(2025, 1, 1)] * len(user_ids) 
        }
    )
    # Select which item-user interactions to use for the training
    item_user_interactions_df = pd.read_parquet('./feature_repo/data/interactions_item_user_ids.parquet')
    item_user_neg_interactions_df = pd.read_parquet('./feature_repo/data/neg_interactions_item_user_ids.parquet')
    item_user_interactions_df['event_timestamp'] = datetime(2025, 1, 1)
    item_user_neg_interactions_df['event_timestamp'] = datetime(2025, 1, 1)

    # retrive datasets for training
    item_df = store.get_historical_features(entity_df=item_entity_df, features=item_service).to_df()
    user_df = store.get_historical_features(entity_df=user_entity_df, features=user_service).to_df()
    interaction_df = store.get_historical_features(entity_df=item_user_interactions_df, features=interaction_service).to_df()
    neg_interaction_df = store.get_historical_features(entity_df=item_user_neg_interactions_df, features=neg_interactions_service).to_df()
    
    # Pass artifacts
    item_df.to_parquet(item_df_output.path)
    user_df.to_parquet(user_df_output.path)
    interaction_df.to_parquet(interaction_df_output.path)
    neg_interaction_df.to_parquet(neg_interaction_df_output.path)
    
    item_df_output.metadata['format'] = 'parquet'
    user_df_output.metadata['format'] = 'parquet'
    interaction_df_output.metadata['format'] = 'parquet'
    neg_interaction_df_output.metadata['format'] = 'parquet'

    
@dsl.pipeline(name=os.path.basename(__file__).replace(".py", ""))
def batch_recommendation():
    
    load_data_task = load_data_from_feast()
    # Component configurations
    load_data_task.set_caching_options(False)
    
    train_model_task = train_model(
        item_df_input=load_data_task.outputs['item_df_output'],
        user_df_input=load_data_task.outputs['user_df_output'],
        interaction_df_input=load_data_task.outputs['interaction_df_output'],
        neg_interaction_df_input=load_data_task.outputs['neg_interaction_df_output'],
    ).after(load_data_task)
    train_model_task.set_caching_options(False)
    
    generate_candidates_task = generate_candidates(
        item_input_model=train_model_task.outputs['item_output_model'],
        user_input_model=train_model_task.outputs['user_output_model'],
        item_df_input=load_data_task.outputs['item_df_output'],
        user_df_input=load_data_task.outputs['user_df_output'],
    ).after(train_model_task)
    generate_candidates_task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=batch_recommendation,
        package_path=__file__.replace(".py", ".yaml"),
    )