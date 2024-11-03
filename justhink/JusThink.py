from flask.cli import load_dotenv
from openai import OpenAI
import json
import logging
import tiktoken
import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError
from tqdm import tqdm
import time
import signal
import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import random
from queue import PriorityQueue
import re
import signal
import concurrent.futures
from threading import Lock
from functools import lru_cache
import math
from datetime import datetime, timedelta  # Added for temporal rewards
import threading


# os.environ['ENV'] = 'local'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analysis.log")
    ]
)

OPENAPI_KEY = os.getenv()

client = OpenAI(api_key=OPENAPI_KEY)

    
class GracefulKiller:
    kill_now = False
    def __init__(self):
        # signal.signal(signal.SIGINT, self.exit_gracefully)
        # signal.signal(signal.SIGTERM, self.exit_gracefully)
        return

    def exit_gracefully(self, signum, frame):
        if self.kill_now == True:
            self.kill_now = True

class ExecutorModuleRules:
    def __init__(self, rules_context, decision_module_rules):
        self.rules_context = rules_context
        self.cache = {}  # Add cache dictionary
        self.decision_module_rules = decision_module_rules  # Reference to update rule scores

    def fetch_rules(self, rules_list):
        logging.info(f"Fetching rules: {rules_list}")
        # Check if the rules are in the cache
        cache_key = rules_list
        if cache_key in self.cache:
            logging.info("Using cached fetched rules.")
            return self.cache[cache_key]

        fetched_rules = {}
        if not rules_list or rules_list.lower() == "none":
            logging.info("No rules to fetch.")
            return fetched_rules
        for rule_number in rules_list.split(','):
            rule_number = rule_number.strip()
            rule_info = self.rules_context.get(rule_number)
            if isinstance(rule_info, dict) and 'rule_content' in rule_info:
                fetched_rules[rule_number] = rule_info['rule_content']
            else:
                logging.warning(f"Rule number {rule_number} not found in rules context or is improperly structured.")

        self.cache[cache_key] = fetched_rules  # Cache
        logging.info(f"Fetched rules: {json.dumps(fetched_rules, indent=2)}")
        return fetched_rules

    def update_rules_based_on_rca(self, rules_in_rca):
        """
        Update the rule scores based on their involvement in RCAs.
        """
        self.decision_module_rules.update_rule_scores(rules_in_rca)

class VectorSearch:
    def __init__(self, embedding_model="text-embedding-ada-002"):
        self.embedding_model = embedding_model
        self.embedding_dim = 1536 if "ada-002" in embedding_model else 768
        self.rule_embeddings = None
        self.rule_ids = []
        self.rule_nn = None
        self.field_embeddings = {}
        self.field_ids = {}
        self.field_nn = {}
        self.killer = GracefulKiller()
        self.embeddings_dir = 'embeddings_cache'  # Directory to store embeddings

        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)
        # Tokenizer for the specific model
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def load_embeddings(self, file_name):
        """Load embeddings from file if it exists."""
        file_path = os.path.join(self.embeddings_dir, file_name)
        if os.path.exists(file_path):
            logging.info(f"Loading embeddings from {file_path}")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            logging.info(f"Embeddings file {file_path} not found.")
            return None

    def save_embeddings(self, file_name, embeddings_data):
        """Save embeddings to file."""
        file_path = os.path.join(self.embeddings_dir, file_name)
        logging.info(f"Saving embeddings to {file_path}")
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings_data, f)

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10))
    def generate_embedding(self, text):
        try:
            logging.info(f"Generating embedding for text: {text[:50]}...")
            response = client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            time.sleep(0.1)  # Small delay between requests
            logging.info("Embedding generated successfully")
            return embedding
        except Exception as e:
            logging.error(f"Failed to generate embedding for text: {text[:50]}. Error: {e}")
            raise

    def build_rule_vectors(self, rules_context):
        embeddings_file = 'rule_embeddings.pkl'

        # Load previously saved embeddings if available
        saved_data = self.load_embeddings(embeddings_file)
        if saved_data:
            self.rule_embeddings, self.rule_ids = saved_data
            self.rule_nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(self.rule_embeddings)
            logging.info("Rule embeddings loaded from file successfully.")
            return

        logging.info("Starting to build rule vectors...")
        self.rule_ids = list(rules_context.keys())
        logging.info(f"Total rules to process: {len(self.rule_ids)}")
        embeddings = []
        batch_size = 100
        checkpoint_file = "rule_embeddings_checkpoint.pkl"

        # Load checkpoint if exists
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                embeddings, processed_rules = pickle.load(f)
            logging.info(f"Loaded checkpoint. Resuming from rule {len(processed_rules)}")
        else:
            processed_rules = set()

        for i in range(0, len(self.rule_ids), batch_size):
            if self.killer.kill_now:
                logging.info("Received termination signal. Saving progress and exiting.")
                break

            batch = self.rule_ids[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{len(self.rule_ids)//batch_size + 1}")
            for rule_id in tqdm(batch, desc=f"Generating embeddings for batch {i//batch_size + 1}"):
                if rule_id in processed_rules:
                    continue
                try:
                    rule_info = rules_context.get(rule_id)
                    if not isinstance(rule_info, dict):
                        logging.error(f"Rule ID {rule_id} is not a dictionary: {rule_info}")
                        embeddings.append([0.0]*self.embedding_dim)
                        continue
                    if "context_description" not in rule_info:
                        logging.error(f"Rule ID {rule_id} is missing 'context_description': {rule_info}")
                        embeddings.append([0.0]*self.embedding_dim)
                        continue
                    embedding = self.generate_embedding(rule_info["context_description"])
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        embeddings.append([0.0]*self.embedding_dim)
                    processed_rules.add(rule_id)
                except RetryError:
                    logging.error(f"All retry attempts failed for rule ID {rule_id}. Assigning zero vector.")
                    embeddings.append([0.0]*self.embedding_dim)
                except Exception as e:
                    logging.error(f"Error processing rule ID {rule_id}: {e}")
                    embeddings.append([0.0]*self.embedding_dim)

            # Save checkpoint after each batch
            with open(checkpoint_file, 'wb') as f:
                pickle.dump((embeddings, processed_rules), f)

        self.rule_embeddings = np.array(embeddings)
        self.rule_nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(self.rule_embeddings)
        logging.info("Rule embeddings built successfully.")

        # Save the final embeddings and remove the checkpoint file
        self.save_embeddings(embeddings_file, (self.rule_embeddings, self.rule_ids))
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

    def build_field_vectors(self, fields_context):
        logging.info("Starting to build field vectors...")

        for json_file, fields in fields_context.items():
            embeddings_file = f'{json_file}_embeddings.pkl'

            # Load previously saved embeddings if available
            saved_data = self.load_embeddings(embeddings_file)
            print(saved_data)
            if saved_data:
                self.field_embeddings[json_file], self.field_ids[json_file] = saved_data
                self.field_nn[json_file] = NearestNeighbors(n_neighbors=10, metric='cosine').fit(self.field_embeddings[json_file])
                logging.info(f"Field embeddings for {json_file} loaded from file successfully.")
                continue

            logging.info(f"Processing fields for {json_file}")
            
            self.field_ids[json_file] = list(fields.keys())
            embeddings = []
            checkpoint_file = f"{json_file}_embeddings_checkpoint.pkl"

            # Load checkpoint if exists
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    embeddings, processed_fields = pickle.load(f)
                logging.info(f"Loaded checkpoint for {json_file}. Resuming from field {len(processed_fields)}")
            else:
                processed_fields = set()

            for field_id in tqdm(self.field_ids[json_file], desc=f"Generating embeddings for {json_file}"):
                if self.killer.kill_now:
                    logging.info("Received termination signal. Saving progress and exiting.")
                    break

                if field_id in processed_fields:
                    continue
                try:
                    logging.info(f"Starting embedding generation for field {field_id}")
                    embedding = self.generate_embedding(fields[field_id])
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        embeddings.append([0.0]*self.embedding_dim)
                    processed_fields.add(field_id)
                    logging.info(f"Finished embedding generation for field {field_id}")

                    # Save checkpoint every 10 fields
                    if len(processed_fields) % 10 == 0:
                        with open(checkpoint_file, 'wb') as f:
                            pickle.dump((embeddings, processed_fields), f)
                except RetryError:
                    logging.error(f"All retry attempts failed for field ID {field_id} in {json_file}. Assigning zero vector.")
                    embeddings.append([0.0]*self.embedding_dim)
                except Exception as e:
                    logging.error(f"Error processing field ID {field_id} in {json_file}: {e}")
                    embeddings.append([0.0]*self.embedding_dim)

            self.field_embeddings[json_file] = np.array(embeddings)
            self.field_nn[json_file] = NearestNeighbors(n_neighbors=10, metric='cosine').fit(self.field_embeddings[json_file])

            # Save the final embeddings and remove the checkpoint file
            self.save_embeddings(embeddings_file, (self.field_embeddings[json_file], self.field_ids[json_file]))
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

        logging.info("Field embeddings built successfully.")

    def load_or_build_vectors(self, rules_context, fields_context):
        # Check and build rule embeddings if they are not already built
        if self.rule_embeddings is None:
            self.build_rule_vectors(rules_context)
        else:
            logging.info("Rule embeddings already built.")

        # Check and build field embeddings if they are not already built

        fields_to_build = []
        for json_file in fields_context:
            if json_file not in self.field_embeddings or self.field_embeddings[json_file].size == 0:
                fields_to_build.append(json_file)
        if fields_to_build:
            self.build_field_vectors({k: fields_context[k] for k in fields_to_build})
        else:
            logging.info("Field embeddings already built.")

    def search_rules(self, thought, top_k=5):
        logging.info(f"Searching rules for thought: {thought[:50]}...")
        try:
            thought_embedding = self.generate_embedding(thought)
        except RetryError:
            logging.error("All retry attempts failed for generating embedding of thought.")
            return []
        except Exception as e:
            logging.error(f"Error generating embedding for thought: {e}")
            return []

        if thought_embedding is None:
            logging.warning("Failed to generate embedding for thought. Returning empty list.")
            return []
        thought_embedding = np.array(thought_embedding).reshape(1, -1)
        distances, indices = self.rule_nn.kneighbors(thought_embedding, n_neighbors=top_k)
        relevant_rule_ids = [self.rule_ids[idx] for idx in indices[0]]
        logging.info(f"Found {len(relevant_rule_ids)} relevant rules.")
        return relevant_rule_ids

    def search_fields_based_on_rules(self, fetched_rule_ids, top_k=10):
        """
        For each fetched rule, find relevant fields by computing vector similarity
        between the rule's embedding and the field context embeddings.
        Aggregate the top fields across all rules.
        """
        logging.info("Searching fields based on fetched rules...")
        relevant_fields = {}

        # Aggregate embeddings of fetched rules
        rule_embeddings = []
        for rule_id in fetched_rule_ids:
            try:
                idx = self.rule_ids.index(rule_id)
                rule_embeddings.append(self.rule_embeddings[idx])
            except ValueError:
                logging.error(f"Rule ID {rule_id} not found in rule_ids.")
                continue
        if not rule_embeddings:
            logging.warning("No valid rule embeddings found for the fetched rules.")
            return relevant_fields

        rule_embeddings = np.array(rule_embeddings)

        # Compute average embedding for the fetched rules
        average_rule_embedding = np.mean(rule_embeddings, axis=0).reshape(1, -1)

        # For each context file, perform similarity search
        for json_file, nn_model in self.field_nn.items():
            if nn_model is None:
                logging.warning(f"No NearestNeighbors model found for {json_file}. Skipping.")
                continue
            distances, indices = nn_model.kneighbors(average_rule_embedding, n_neighbors=top_k)
            top_field_ids = [self.field_ids[json_file][idx] for idx in indices[0]]
            relevant_fields[json_file] = top_field_ids

        logging.info(f"Found relevant fields based on rules: {relevant_fields}")
        return relevant_fields

    def search_fields(self, thought, json_file, top_k=10):
        logging.info(f"Searching fields in {json_file} for thought: {thought[:50]}...")
        if json_file not in self.field_nn:
            logging.warning(f"No vector index found for {json_file}")
            return []
        try:
            thought_embedding = self.generate_embedding(thought)
        except RetryError:
            logging.error("All retry attempts failed for generating embedding of thought.")
            return []
        except Exception as e:
            logging.error(f"Error generating embedding for thought: {e}")
            return []

        if thought_embedding is None:
            logging.warning("Failed to generate embedding for thought. Returning empty list.")
            return []
        thought_embedding = np.array(thought_embedding).reshape(1, -1)
        distances, indices = self.field_nn[json_file].kneighbors(thought_embedding, n_neighbors=top_k)
        relevant_field_ids = [self.field_ids[json_file][idx] for idx in indices[0]]
        logging.info(f"Found {len(relevant_field_ids)} relevant fields in {json_file}.")
        return relevant_field_ids

# Data Loader Module
class DataLoader:
    def __init__(self, log, merchant_details, transaction_details):
        logging.info("Initializing DataLoader")
        load_dotenv()
        self.environment = os.getenv("ENV")

        self.log_json = self.load(log)
        self.transaction_meta_json = self.load(transaction_details)
        self.merchant_config_json = self.load(merchant_details)
        self.context_text = self.load_text('Context.txt')
        self.rules_json = None
        self.rules_context = None

        # print("---------------------------------log json-----------------------------------")
        # print(self.log_json)
        # print("----------------------------------------------------------------------------------------")
        # print("---------------------------------transa json-----------------------------------")
        # print(self.transaction_meta_json)
        # print("----------------------------------------------------------------------------------------")
        # print("---------------------------------log json-----------------------------------")
        # print(self.merchant_config_json)
        # print("----------------------------------------------------------------------------------------")

        # Extract field names
        self.log_fields = self.extract_fields(self.log_json)
        self.transaction_meta_fields = self.extract_fields(self.transaction_meta_json)
        self.merchant_config_fields = self.extract_fields(self.merchant_config_json)

        # Load or generate context for fields (using context files)
        self.fields_context = {
            'Log.json': self.load_or_generate_field_context('Log.json', self.log_json),
            'Transaction Meta Data.json': self.load_or_generate_field_context('Transaction Meta Data.json', self.transaction_meta_json),
            'Merchant Configurations.json': self.load_or_generate_field_context('Merchant Configurations.json', self.merchant_config_json),
        }

    def load(self, data):
        if self.environment == "PROD":
            return self._load_json_prod(data)
        elif self.environment == "DEV":
            return self._load_json_dev(data)
        return self._load_json_prod(self, data)
        
    @staticmethod
    def _load_json_prod(self,data):
        logging.info("Loading JSON data.")
        try:
            # If data is a string, try to load it as JSON
            if isinstance(data, str):
                loaded_data = json.loads(data)
                logging.info("Successfully loaded JSON data from string.")
                return loaded_data
            
            # If data is already a dictionary, just return it
            elif isinstance(data, dict):
                logging.info("Received data is already a dictionary.")
                return data
            
            else:
                logging.error("Invalid data type. Expected string or dictionary.")
                return {}
        
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return {}

    @staticmethod
    def _load_json_dev(file_path):
        logging.info(f"Loading JSON file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                logging.info(f"Successfully loaded {file_path}")
                return data
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error in {file_path}: {e}")
            return {}

    @staticmethod
    def load_text(file_path):
        logging.info(f"Loading text file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = f.read()
                logging.info(f"Successfully loaded {file_path}")
                return data
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return ""
        except Exception as e:
            logging.error(f"Error loading text file {file_path}: {e}")
            return ""

    @staticmethod
    def extract_fields(json_data, parent_key=''):
        fields = []
        for key, value in json_data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                fields.extend(DataLoader.extract_fields(value, full_key))
            else:
                fields.append(full_key)
        return fields

    def load_or_generate_field_context(self, json_file, json_data):
        context_file = f"{json_file}_context.json"
        if os.path.exists(context_file):
            logging.info(f"Loading existing context for {json_file}")
            try:
                with open(context_file, 'r') as f:
                    context = json.load(f)
                return context
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in {context_file}: {e}")
                logging.info(f"Regenerating context for {json_file}")
        else:
            logging.info(f"Generating context for {json_file}")

        context = {}
        for field in self.extract_fields(json_data):
            context[field] = self.generate_field_context(json_file, field)
        try:
            with open(context_file, 'w') as f:
                json.dump(context, f, indent=2)
            logging.info(f"Context for {json_file} saved to {context_file}")
        except Exception as e:
            logging.error(f"Error saving context for {json_file}: {e}")
        return context

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10))
    def generate_field_context(self, json_file, field):
        prompt = f"""
You are an expert in payment systems. Provide a concise description of the following field from {json_file}:

Field: "{field}"

Description:
"""
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=100,
                temperature=0.2,
            )
            description = response.choices[0].text.strip()
            logging.info(f"Generated context for field '{field}': {description}")
            return description
        except Exception as e:
            logging.error(f"Error generating context for field '{field}': {e}")
            raise

    def convert_context_to_json_rules(self):
        rules_json_file = 'rules.json'
        rules_context_file = 'rules_context.json'

        # Check if both files exist and are valid
        if os.path.exists(rules_json_file) and os.path.exists(rules_context_file):
            try:
                with open(rules_json_file, 'r') as f:
                    self.rules_json = json.load(f)
                with open(rules_context_file, 'r') as f:
                    self.rules_context = json.load(f)
                logging.info("Rules and rules context loaded successfully. Skipping conversion.")
                return self.rules_context
            except json.JSONDecodeError as e:
                logging.warning(f"Existing rules or rules context files are corrupted: {e}. Regenerating them.")

        logging.info("Converting context to JSON rules")
        prompt = f"""
You are an expert in converting natural language descriptions into structured JSON rules.

Convert the following context into a JSON format where each rule is an object with a "rule_number" and "rule_content":

{self.context_text}

Output the rules in valid JSON format.
"""

        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1500,
                temperature=0,
            )
            self.rules_json = json.loads(response.choices[0].text.strip())

            # Load or generate context for rules
            self.rules_context = self.load_or_generate_rules_context()

            with open(rules_json_file, 'w') as f:
                json.dump(self.rules_json, f, indent=2)

            with open(rules_context_file, 'w') as f:
                json.dump(self.rules_context, f, indent=2)

            logging.info("Context converted to JSON rules and saved to rules.json and rules_context.json")
            return self.rules_context
        except Exception as e:
            logging.error(f"Error converting context to JSON rules: {e}")
            self.rules_json = []
            return {}

    def load_or_generate_rules_context(self):
        context_file = "rules_context.json"
        if os.path.exists(context_file):
            logging.info("Loading existing context for rules")
            try:
                with open(context_file, 'r') as f:
                    rules_context = json.load(f)
                # Validate the structure of rules_context
                for rule_id, rule_info in rules_context.items():
                    if not isinstance(rule_info, dict):
                        raise ValueError(f"Rule ID {rule_id} does not map to a dictionary.")
                    if "rule_content" not in rule_info or "context_description" not in rule_info:
                        raise ValueError(f"Rule ID {rule_id} is missing required keys.")
                return rules_context
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Error loading rules context: {e}")
                logging.info("Regenerating context for rules.")
        else:
            logging.info("Generating context for rules")

        rules_context = {}
        for rule in self.rules_json:
            rule_number = rule.get('rule_number')
            rule_content = rule.get('rule_content', '')
            context = self.generate_rule_context(rule_number, rule_content)
            if rule_number is not None:
                rules_context[str(rule_number)] = {
                    "rule_content": rule_content,
                    "context_description": context
                }
            else:
                logging.warning(f"Rule without a number encountered: {rule}")

        try:
            with open(context_file, 'w') as f:
                json.dump(rules_context, f, indent=2)
            logging.info("Context for rules saved to rules_context.json")
        except Exception as e:
            logging.error(f"Error saving context for rules: {e}")
        return rules_context

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=10))
    def generate_rule_context(self, rule_number, rule_content):
        prompt = f"""
You are an expert in payment systems. Provide a concise description of the following rule:

Rule Number: {rule_number}
Rule Content: "{rule_content}"

Description:
"""
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=150,
                temperature=0.2,
            )
            description = response.choices[0].text.strip()
            logging.info(f"Generated context for rule '{rule_number}': {description}")
            return description
        except Exception as e:
            logging.error(f"Error generating context for rule '{rule_number}': {e}")
            raise

    def get_field_value(self, json_file, field_path):
        """
        Retrieves the value of a nested field from a specified JSON file.
        """
        json_mapping = {
            'log': self.log_json,
            'transaction_meta_data': self.transaction_meta_json,
            'merchant_configurations': self.merchant_config_json,
        }
        data = json_mapping.get(json_file.lower())
        if not data:
            logging.error(f"Unknown JSON file: {json_file}")
            return None

        keys = field_path.split('.')
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key)
                if data is None:
                    logging.warning(f"Field '{field_path}' not found in {json_file}.")
                    return None
            else:
                logging.warning(f"Field '{field_path}' is not a dictionary in {json_file}.")
                return None
        return data

class ThoughtNode:
    def __init__(self, description, parent=None, depth=0):
        self.description = description
        self.parent = parent
        self.children = []
        self.fetched_data = {}
        self.fetched_rules = {}
        self.q_value = 1.0  # Optimistically initialized Q-value
        self.target_q_value = 1.0  # Target Q-value for stabilization
        self.heuristic = 0
        self.depth = depth
        self.visits = 0
        self.rca_possible = None  # New attribute to store RCA possibility
        self.rca_confidence = 0   # Confidence level for RCA possibility
        self.timestamp = datetime.now()  # Added for temporal rewards

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_unique_id(self):
        path = []
        current = self
        while current:
            path.append(current.description)
            current = current.parent
        return '->'.join(reversed(path))

    def __lt__(self, other):
        return (self.q_value + self.heuristic) > (other.q_value + other.heuristic)

# Decision Module JSON
class DecisionModuleJSON:
    def __init__(self, data_loader, vector_search):
        self.data_loader = data_loader
        self.vector_search = vector_search

    def decide_fields_based_on_rules(self, fetched_rule_ids):
        logging.info(f"Deciding fields based on fetched rules: {fetched_rule_ids}")

        # Use vector similarity to find relevant fields based on the fetched rules
        relevant_fields = self.vector_search.search_fields_based_on_rules(fetched_rule_ids, top_k=10)

        fields_with_explanation = ""
        for json_file, fields in relevant_fields.items():
            if fields:
                fields_str = ', '.join(fields)
                fields_with_explanation += f"{json_file}: {fields_str}\nReason: Relevant to the fetched rules.\n\n"
            else:
                fields_with_explanation += f"{json_file}: None\nReason: No relevant fields found based on the rules.\n\n"

        logging.info(f"Fields decided with explanation: {fields_with_explanation}")
        return fields_with_explanation

    @staticmethod
    def truncate_text(text, max_length):
        return text[:max_length] + "..." if len(text) > max_length else text

    @staticmethod
    def truncate_json(json_obj, max_length):
        json_str = json.dumps(json_obj)
        if len(json_str) <= max_length:
            return json_obj

        truncated_obj = {}
        current_length = 2  # Count the opening and closing braces

        for key, value in json_obj.items():
            value_str = json.dumps(value)
            if current_length + len(key) + len(value_str) + 5 > max_length:  # 5 accounts for quotes, colon, and comma
                break
            truncated_obj[key] = value
            current_length += len(key) + len(value_str) + 5

        return truncated_obj

# Decision Module Rules
class DecisionModuleRules:
    def __init__(self, rules_context, vector_search):
        self.rules_context = rules_context
        self.vector_search = vector_search
        self.rca_cache = {}  # Cache to store RCA possibility checks
        self.rule_scores = {rule_id: 1.0 for rule_id in rules_context.keys()}  # Initialize rule scores

    def decide_rules(self, node_description):
        logging.info(f"Deciding rules for: {node_description}")

        # Use vector search to find relevant rules based on the node description
        relevant_rule_ids = self.vector_search.search_rules(node_description, top_k=5)
        if not relevant_rule_ids:
            return "None"

        # Sort rules based on their dynamic scores (higher score first)
        sorted_rules = sorted(relevant_rule_ids, key=lambda x: self.rule_scores.get(x, 1.0), reverse=True)
        selected_rules = ','.join(sorted_rules)
        logging.info(f"Rules decided: {selected_rules}")
        return selected_rules

    def update_rule_scores(self, rules_in_rca):
        """
        Update the scores of rules based on their involvement in successful RCAs.
        """
        for rule_id in rules_in_rca:
            if rule_id in self.rule_scores:
                self.rule_scores[rule_id] += 0.1  # Increment score
            else:
                self.rule_scores[rule_id] = 1.1  # Initialize score

    def get_rule_descriptions(self):
        descriptions = []
        for rule_id, rule_info in self.rules_context.items():
            description = f"{rule_id}: {rule_info['rule_content']}"
            descriptions.append(description)
        return "\n".join(descriptions)

# Executor Module JSON
class ExecutorModuleJSON:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.cache = {}  # Cache dictionary
        # Mapping original JSON files for fetching values from them
        self.json_file_mapping = {
            'log.json': 'log',
            'transaction_meta_data.json': 'transaction_meta_data',
            'merchant_configurations.json': 'merchant_configurations',
        }

    def fetch_fields(self, fields_with_explanation):
        logging.info(f"Fetching fields: {fields_with_explanation}")
        # Check if the fields are in the cache
        cache_key = fields_with_explanation
        if cache_key in self.cache:
            logging.info("Using cached fetched data.")
            return self.cache[cache_key]

        fetched_data = {}
        current_json = ""

        # Split lines and remove any empty lines
        lines = [line.strip() for line in fields_with_explanation.strip().split('\n') if line.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if the line is a JSON file line
            if ':' in line and not line.lower().startswith('reason:'):
                try:
                    json_line, fields_str = line.split(':', 1)
                except ValueError:
                    logging.warning(f"Malformed JSON line: {line}")
                    i += 1
                    continue

                current_json = json_line.strip().lower()
                current_json = current_json.replace('.json', '').replace(' ', '_').replace('-', '_')

                json_file_attr = self.json_file_mapping.get(f"{current_json}.json")
                if json_file_attr is None:
                    logging.warning(f"Unknown JSON file: {current_json}")
                    i += 1
                    continue

                fields = [f.strip() for f in fields_str.strip().split(',') if f.strip().lower() != 'none']
                if fields:
                    for field in fields:
                        field = field.strip("[]\"'")
                        # Fetch values from the original JSON files
                        value = self.data_loader.get_field_value(json_file_attr, field)
                        fetched_data[f"{current_json}.json: {field}"] = value

                # Check if the next line exists and is a reason line
                if (i + 1) < len(lines) and lines[i + 1].lower().startswith('reason:'):
                    reason_line = lines[i + 1]
                    try:
                        _, reason = reason_line.split(':', 1)
                        reason = reason.strip()
                        fetched_data[f"{current_json}_reason"] = reason
                    except ValueError:
                        fetched_data[f"{current_json}_reason"] = ""
                    i += 2  # Move past the reason line
                else:
                    fetched_data[f"{current_json}_reason"] = ""
                    i += 1
            else:
                # If the line is a reason line or malformed, skip it
                if line.lower().startswith('reason:'):
                    logging.warning(f"Unexpected reason line without corresponding JSON file: {line}")
                else:
                    logging.warning(f"Malformed or unexpected line: {line}")
                i += 1

        self.cache[cache_key] = fetched_data  # Store in cache
        logging.info(f"Fetched data with explanations: {json.dumps(fetched_data, indent=2)}")
        return fetched_data

# Graph of Thoughts Manager
class GraphOfThoughts:
    def __init__(self, udf_order_id, data_loader, vector_search, config=None):
        logging.info(f"Starting analysis for udfOrderId: {udf_order_id}")
        self.udf_order_id = udf_order_id
        self.data_loader = data_loader
        self.vector_search = vector_search
        self.decision_module_json = DecisionModuleJSON(data_loader, vector_search)
        self.decision_module_rules = DecisionModuleRules(data_loader.rules_context, vector_search)
        self.executor_module_rules = ExecutorModuleRules(data_loader.rules_context, self.decision_module_rules)
        self.executor_module_json = ExecutorModuleJSON(data_loader)
        self.visited_nodes = set()
        self.queue = PriorityQueue()
        self.iteration_limit = config.get('iteration_limit', 1000) if config else 1000
        self.iteration_limit_without_improvement = config.get('iteration_limit_without_improvement', 100) if config else 100
        self.iterations = 0
        self.graph = nx.DiGraph()
        self.root = ThoughtNode(f"Start analysis for order ID {udf_order_id}", depth=0)
        self.graph.add_node(self.root.description, node=self.root)
        self.pos = None
        self.traversed_path = []
        self.recurring_missing_info = set()
        self.recurring_threshold = config.get('recurring_threshold', 3) if config else 3
        self.max_iterations = config.get('max_iterations', 800) if config else 800
        self.max_depth = config.get('max_depth', 25) if config else 25
        self.min_exploration_depth = config.get('min_exploration_depth', 3) if config else 3
        self.rca_confidence_threshold = config.get('rca_confidence_threshold', 0.7) if config else 0.7
        self.convergence_threshold = config.get('convergence_threshold', 0.005) if config else 0.005
        self.breadth = config.get('breadth', 7) if config else 7
        self.visualization_interval = config.get('visualization_interval', 15) if config else 15

        # Initialize dynamic parameters
        self.alpha = 0.3  # Initial learning rate
        self.gamma = 0.8  # Initial discount factor
        self.epsilon = 0.7  # Increased from 0.5 to 0.7 for more exploration
        self.epsilon_min = 0.1  # Minimum epsilon value
        self.epsilon_decay_factor = 0.995  # Slower decay

        # Weights for priority calculation
        self.q_weight = 0.7  # Increased weight for Q-value
        self.h_weight = 0.3  # Decreased weight for heuristic

        # Reward coefficients
        self.data_reward_weight = 0.5  # Increased weight for data reward
        self.rule_reward_weight = 0.6  # Increased weight for rule reward
        self.rca_reward_scale = 20.0   # Increased RCA reward scale
        self.depth_penalty_value = -1.0  # Reduced depth penalty

        # For summary generation
        self.analysis_steps = []

        # Initialize tracking sets for unique data and rules
        self.visited_data = set()
        self.visited_rules = set()

        # Initialize a lock for thread-safe operations
        self.lock = Lock()

        # Initialize RCA cache to reduce redundant API calls
        self.rca_cache = {}

        # To store multiple RCA findings
        self.rca_findings = []

        # Experience replay buffer
        self.experience_replay_buffer = []
        self.replay_buffer_size = config.get('replay_buffer_size', 100)

        # UCB exploration parameter
        self.c = config.get('exploration_constant', 1.9)  # Updated exploration parameter

        # For dynamic parameter adjustment
        self.iterations_without_improvement = 0
        self.performance_history = []

        # Initialize high-value paths
        self.high_value_paths = set()
        self.explored_thoughts = set()

        # Enhanced: Initialize a set to track timestamps for temporal rewards
        self.node_timestamps = {}

        # Initialize target network parameters
        self.target_q_update_frequency = config.get('target_q_update_frequency', 100) if config else 100
        self.iterations_since_target_update = 0  # Counter to track when to update target network

    def compute_priority(self, node):
        # Combine q_value and heuristic with weights
        return -(self.q_weight * node.q_value + self.h_weight * node.heuristic)

    def calculate_heuristic(self, node):
        with self.lock:
            # Calculate relevance scores
            data_relevance = sum(
                1 for key in node.fetched_data if key != 'overall_explanation' and not key.endswith('_reason')
            )
            rule_relevance = len(node.fetched_rules)

            # Information Gain: Prioritize nodes that fetch more unique data and rules
            unique_data = len(set(node.fetched_data.keys()) - self.visited_data)
            unique_rules = len(set(node.fetched_rules.keys()) - self.visited_rules)

            # Update visited data and rules
            self.visited_data.update(node.fetched_data.keys())
            self.visited_rules.update(node.fetched_rules.keys())

        # Depth Factor: Encourage deeper exploration but penalize excessive depth
        depth_penalty = (node.depth / self.max_depth) ** 2  # Quadratic penalty for depth

        # Incorporate the number of unique fields fetched
        unique_fields = len(set(node.fetched_data.keys()))
        field_diversity_score = unique_fields / (node.depth + 1)

        # Include a factor for unexplored nodes
        if node.get_unique_id() not in self.visited_nodes:
            unexplored_bonus = 2.0
        else:
            unexplored_bonus = 0.0

        # Dynamic Criticality Assessment
        # Determine the importance of each fetched rule based on their dynamic scores
        critical_rule_involvement = sum([self.decision_module_rules.rule_scores.get(rule_id, 1.0) for rule_id in node.fetched_rules.keys()])
        context_adjustment = critical_rule_involvement * 0.5  # Scale down to prevent overemphasis

        # Adjusted Heuristic Calculation
        heuristic = (data_relevance + rule_relevance + field_diversity_score) * (1 - depth_penalty) + \
                    (unique_data + unique_rules) * 0.5 + unexplored_bonus + context_adjustment

        return heuristic

    def update_q_value(self, node, reward):
        # Store experience
        self.experience_replay_buffer.append((node, reward))
        if len(self.experience_replay_buffer) > self.replay_buffer_size:
            self.experience_replay_buffer.pop(0)

        # Sample from experience replay buffer
        experiences = random.sample(self.experience_replay_buffer, min(len(self.experience_replay_buffer), 10))
        for exp_node, exp_reward in experiences:
            if exp_node.parent:
                # Use target_q_value for max_child_q
                max_child_q = max([child.target_q_value for child in exp_node.children], default=0)
                exp_node.q_value += self.alpha * (exp_reward + self.gamma * max_child_q - exp_node.q_value)
                logging.debug(f"Updated Q-value for node '{exp_node.description}': {exp_node.q_value}")

    def select_next_node(self):
        total_visits = sum(node.visits for _, node in self.queue.queue) + 1  # Avoid division by zero

        def ucb_score(node):
            if node.visits == 0:
                return float('inf')
            exploitation = node.q_value
            exploration = self.c * math.sqrt(math.log(total_visits) / node.visits)
            return exploitation + exploration

        if not self.queue.empty():
            # Select the node with the highest UCB score
            selected_item = max(self.queue.queue, key=lambda x: ucb_score(x[1]))
            self.queue.queue.remove(selected_item)
            selected_node = selected_item[1]
            logging.debug(f"UCB selected node: {selected_node.description}")
            return selected_node
        return None

    @lru_cache(maxsize=1024)
    def cached_check_rca_possible(self, node_description, fetched_rules_json, fetched_data_json):
        # Create a unique key for caching
        cache_key = (node_description, json.dumps(fetched_rules_json, sort_keys=True), json.dumps(fetched_data_json, sort_keys=True))
        return cache_key

    def fetch_relevant_data(self, node):
        """
        Fetches relevant rules and fields based on the current thought.
        Ensures that only data pertinent to the thought and rules is fetched.
        """
        # Decide which rules are relevant to the current thought
        rules_needed = self.decision_module_rules.decide_rules(node.description)

        # Use ThreadPoolExecutor to fetch rules and data in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:  # Increased workers
            future_rules = executor.submit(self.executor_module_rules.fetch_rules, rules_needed)

            # Decide which fields are needed based on the fetched rules
            if rules_needed != "None":
                fetched_rule_ids = rules_needed.split(',')
                future_fields_with_explanation = executor.submit(
                    self.decision_module_json.decide_fields_based_on_rules, fetched_rule_ids
                )
                fields_with_explanation = future_fields_with_explanation.result()
                future_data = executor.submit(self.executor_module_json.fetch_fields, fields_with_explanation)
                node.fetched_data = future_data.result()
            else:
                node.fetched_data = {}

            node.fetched_rules = future_rules.result()

        # Log the fetched data and rules for summary
        step_detail = {
            "Thought": node.description,
            "Fetched Rules": node.fetched_rules,
            "Fetched Data": node.fetched_data
        }
        self.analysis_steps.append(step_detail)

    def check_if_rca_possible(self, node):
        """
        Uses the LLM to determine whether the current data is sufficient to make an RCA.
        Implements caching to reduce redundant API calls.
        """
        if node.rca_possible is not None:
            return node.rca_possible

        # Check cache first
        cache_key = self.cached_check_rca_possible(
            node.description,
            json.dumps(node.fetched_rules, sort_keys=True),
            json.dumps(node.fetched_data, sort_keys=True)
        )
        if cache_key in self.rca_cache:
            node.rca_possible, node.rca_confidence = self.rca_cache[cache_key]
            logging.info("RCA possibility retrieved from cache.")
            return node.rca_possible

        logging.info(f"Checking if RCA is possible for node: {node.description}")
        prompt = f"""
You are an expert in payment systems analysis.

Based on the current thought, fetched rules, and fetched data, determine whether there is enough information to make a root cause analysis.

Current thought:
"{node.description}"

Fetched Rules:
{json.dumps(node.fetched_rules, indent=2)}

Fetched Data:
{json.dumps(node.fetched_data, indent=2)}

Is there enough information to make a Root Cause Analysis? Answer "YES" or "NO" and provide a brief justification.

Also, provide a confidence score between 0 and 1 indicating how confident you are that the information is sufficient, where 1 means absolutely certain and 0 means not at all certain.

Your response should be in the following format:

Answer: [YES/NO]
Confidence: [confidence score]
Justification: [brief justification]
"""
        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=150,
                temperature=0,
            )
            answer = response.choices[0].text.strip()
            logging.info(f"LLM response for RCA possibility:\n{answer}")
            match = re.search(r'Answer:\s*(YES|NO)', answer, re.IGNORECASE)
            confidence_match = re.search(r'Confidence:\s*([0-9]*\.?[0-9]+)', answer)
            if match and confidence_match:
                node.rca_possible = True if match.group(1).strip().upper() == 'YES' else False
                node.rca_confidence = float(confidence_match.group(1))
            else:
                node.rca_possible = False
                node.rca_confidence = 0
            # Update cache
            self.rca_cache[cache_key] = (node.rca_possible, node.rca_confidence)
            return node.rca_possible
        except Exception as e:
            logging.error(f"Error checking RCA possibility: {e}")
            node.rca_possible = False
            node.rca_confidence = 0
            return False

    def expand_node(self, node):
        logging.info(f"Expanding node: {node.description} at depth {node.depth}")
        node.visits += 1  # Increment visit count
        node_id = node.get_unique_id()

        # Enhanced Cycle Detection: Check if node has been visited before
        if node_id in self.visited_nodes:
            logging.info("Cycle detected. Skipping node expansion.")
            return
        self.visited_nodes.add(node_id)

        self.fetch_relevant_data(node)

        # Collect RCA findings from multiple nodes
        if self.check_if_rca_possible(node) and node.rca_confidence >= self.rca_confidence_threshold:
            logging.info("RCA criteria met with sufficient confidence for this node.")
            self.rca_findings.append(node)
            self.executor_module_rules.update_rules_based_on_rca(node.fetched_rules.keys())  # Update rule scores
            self.high_value_paths.add(node.description)  # Add to high-value paths
            # Continue exploration to find more issues
        elif node.depth >= self.min_exploration_depth and self.check_if_rca_possible(node):
            # If RCA is possible but confidence is low, continue exploration
            if node.rca_confidence < self.rca_confidence_threshold:
                logging.info("RCA possible but confidence is low; continuing exploration.")

        next_thoughts = self.generate_next_thoughts(node)

        for thought_description in next_thoughts:
            child_node = ThoughtNode(thought_description, parent=node, depth=node.depth + 1)

            # Enhanced: Check for cycles before adding child
            child_id = child_node.get_unique_id()
            if child_id in self.visited_nodes:
                logging.info(f"Cycle detected for child node '{child_node.description}'. Skipping addition.")
                continue

            node.add_child(child_node)
            child_node.heuristic = self.calculate_heuristic(child_node)
            priority = self.compute_priority(child_node)
            self.queue.put((priority, child_node))
            self.graph.add_node(child_node.description, node=child_node)
            self.graph.add_edge(node.description, child_node.description)
            logging.info(f"Added child node: {child_node.description}")

        self.traversed_path.append(node.description)
        logging.debug(f"Current Traversal Path: {self.traversed_path}")

    def generate_next_thoughts(self, node):
        prompt_template = """
You are an expert in payment systems analysis. Based on the current thought, the fetched rules, and the data fetched according to those rules, analyze what information is still missing to make a root cause analysis, and plan the next steps to obtain that information.

Current thought:
"{thought}"

Fetched Rules:
{rules}

Fetched Data (based on rules):
{data}

Current depth in analysis: {depth}

Generate {num_thoughts} diverse and specific next thoughts to proceed with the analysis. Each thought should be a unique action or analysis step to obtain missing information or further narrow down the root cause. Avoid generic statements and ensure each thought is distinct from the others.
    
Example output:
- Thought 1: Examine the specific error code in the transaction log to understand the failure reason.
- Thought 2: Verify the merchant configuration for the transaction type to check for any misconfigurations.
- Thought 3: Analyze the timing of events in the transaction flow to identify potential bottlenecks.
- Thought 4: Investigate any recent changes in the payment gateway settings that might affect this transaction.
"""
        max_total_tokens = 4097  # Model's maximum context length
        max_completion_tokens = 500
        max_prompt_tokens = max_total_tokens - max_completion_tokens

        thought = self.truncate_text(node.description, 200)
        rules = self.truncate_json(node.fetched_rules, (max_prompt_tokens - 200) // 2)
        data = self.truncate_json(node.fetched_data, (max_prompt_tokens - 200) // 2)

        prompt = prompt_template.format(
            thought=thought,
            rules=json.dumps(rules, indent=2),
            data=json.dumps(data, indent=2),
            depth=node.depth,
            num_thoughts=self.breadth
        )

        prompt_tokens = self.vector_search.count_tokens(prompt)
        if prompt_tokens > max_prompt_tokens:
            prompt = self.truncate_text(prompt, max_prompt_tokens)

        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=max_completion_tokens,
                temperature=0,
            )
            next_thoughts_text = response.choices[0].text.strip()
            logging.debug(f"Received next thoughts from API: {next_thoughts_text}")
        except Exception as e:
            logging.error(f"Error in API call: {e}")
            next_thoughts_text = "Unable to generate next thoughts due to an error."

        next_thoughts = []
        for line in next_thoughts_text.split('\n'):
            line = line.strip()
            if line.startswith('- Thought'):
                thought_description = line.split(':', 1)[1].strip() if ':' in line else line.replace('- Thought', '').strip()
                if thought_description and self.is_relevant_thought(thought_description, node) and thought_description not in next_thoughts:
                    next_thoughts.append(thought_description)

        if not next_thoughts:
            next_thoughts.append("Review current data for overlooked details")

        # Diverse Sampling: Ensure diversity by limiting similar thoughts
        diverse_thoughts = self.ensure_diversity(next_thoughts, node)
        return diverse_thoughts[:self.breadth]  # Limit to the configured number of thoughts

    def ensure_diversity(self, thoughts, node):
        """
        Ensures that the generated thoughts are diverse by checking for semantic differences.
        """
        unique_thoughts = []
        for thought in thoughts:
            if thought not in unique_thoughts:
                unique_thoughts.append(thought)
        return unique_thoughts

    def is_relevant_thought(self, thought, node):
        # Enhanced relevance check based on semantic similarity
        relevant_keywords = set(node.description.lower().split() +
                                [k.lower() for k in node.fetched_data.keys()] +
                                [r.lower() for r in node.fetched_rules.values()])
        thought_words = set(thought.lower().split())
        intersection = len(thought_words.intersection(relevant_keywords))
        return intersection > 1  # Reduced threshold for better inclusivity

    def calculate_reward(self, node):
        # Multi-Dimensional Rewards
        # Reward for fetched unique data and rules
        unique_data = len(set(node.fetched_data.keys()) - self.visited_data)
        unique_rules = len(set(node.fetched_rules.keys()) - self.visited_rules)
        data_reward = unique_data * self.data_reward_weight
        rule_reward = unique_rules * self.rule_reward_weight

        # Additional reward for fetching more data fields
        data_fields_count = len(node.fetched_data)
        data_fields_reward = data_fields_count * 0.1  # New reward component

        # Additional reward if the node meets RCA criteria with high confidence
        if node.rca_possible and node.rca_confidence >= self.rca_confidence_threshold:
            rca_reward = self.rca_reward_scale * node.rca_confidence
        else:
            rca_reward = 0.0

        # Penalty for shallow depth RCA conclusions
        depth_penalty = self.depth_penalty_value if node.depth < self.min_exploration_depth else 0.0

        # Bonus for nodes at optimal depth
        optimal_depth = self.min_exploration_depth + 2
        if node.depth == optimal_depth:
            bonus = 1.0
        else:
            bonus = 0.0

        # Additional reward for revisiting valuable paths
        if node.description in [n.description for n in node.parent.children[:-1]] if node.parent else []:
            repetition_penalty = -0.5
        else:
            repetition_penalty = 0

        # Bonus for exploring new areas
        novelty_bonus = 0.5 if node.description not in self.visited_nodes else 0

        # Temporal Rewards: Decay rewards based on the age of the node
        time_since_creation = datetime.now() - node.timestamp
        decay_factor = max(0.1, 1 - (time_since_creation.total_seconds() / 3600))  # Decay over 1 hour
        temporal_reward = (data_reward + rule_reward + rca_reward + bonus + novelty_bonus) * decay_factor

        # Final reward calculation
        total_reward = temporal_reward + data_fields_reward + depth_penalty + repetition_penalty

        return total_reward

    def truncate_json(self, data, max_length):
        json_str = json.dumps(data, indent=2)
        if len(json_str) <= max_length:
            return data

        # Truncate without breaking JSON
        truncated_obj = {}
        current_length = 2  # Count the opening and closing braces

        for key, value in data.items():
            value_str = json.dumps(value, indent=2)
            if current_length + len(key) + len(value_str) + 5 > max_length:  # 5 accounts for quotes, colon, and comma
                break
            truncated_obj[key] = value
            current_length += len(key) + len(value_str) + 5

        return truncated_obj

    @staticmethod
    def truncate_text(text, max_length):
        return text[:max_length] + "..." if len(text) > max_length else text

    def adjust_epsilon(self, iterations_without_improvement):
        # Adaptive epsilon adjustment based on iterations without improvement
        if iterations_without_improvement > 0 and iterations_without_improvement % 10 == 0:
            # Every 10 iterations without improvement, increase epsilon
            old_epsilon = self.epsilon
            self.epsilon = min(1.0, self.epsilon + 0.05)  # Smaller increment for smoother adjustment
            logging.debug(f"Increased epsilon from {old_epsilon} to {self.epsilon} due to lack of improvement.")
        else:
            # Otherwise, decay epsilon
            old_epsilon = self.epsilon
            self.epsilon = max(0.1, self.epsilon * self.epsilon_decay_factor)
            logging.debug(f"Decayed epsilon from {old_epsilon} to {self.epsilon}")

    def generate_aggregated_rca(self):
        logging.info("Generating aggregated RCA based on multiple findings.")
        # Collect all analysis steps from RCA findings
        aggregated_analysis = {
            "Thoughts": set(),
            "Fetched Rules": set(),
            "Fetched Data": set()
        }
        for node in self.rca_findings:
            aggregated_analysis["Thoughts"].add(f"- {node.description}")
            for rule_id, rule_content in node.fetched_rules.items():
                aggregated_analysis["Fetched Rules"].add(f"Rule {rule_id}: {rule_content}")
            for field, value in node.fetched_data.items():
                aggregated_analysis["Fetched Data"].add(f"{field}: {value}")

        # Convert sets to sorted strings
        aggregated_analysis = {k: '\n'.join(sorted(v)) for k, v in aggregated_analysis.items()}

        prompt = f"""
You are an expert in payment systems analysis. Based on the aggregated analysis below, provide a comprehensive Root Cause Analysis with confidence scores.

**Order ID:** {self.udf_order_id}

**Thought Process:**
{aggregated_analysis['Thoughts']}

**Fetched Rules:**
{aggregated_analysis['Fetched Rules']}

**Fetched Data:**
{aggregated_analysis['Fetched Data']}

**Instructions:**
1. **Root Cause Analysis:** Identify all root causes by referencing specific rules and data points. Explain how each referenced item contributes to the conclusions. Provide confidence scores between 0 and 1 indicating your certainty for each identified root cause.

2. **Suggestions:** Provide actionable steps to resolve each identified issue. Include confidence scores for each suggestion.

3. **Attribution:** Determine whether each issue stems from an external factor (Merchant or Gateway), an internal factor (Juspay), or if it's ambiguous based on the available data. Provide confidence scores for each attribution.

4. **Attribution Tags:** Provide one of the following tags for each issue: MERCHANT | GATEWAY | JUSPAY | AMBIGUOUS.

**Please follow the above structure and level of detail in your response, including confidence scores for each section.**

**Format your response as follows:**

Issue 1:
Root Cause Analysis: [Your detailed analysis here] Confidence: [score]

Suggestion: [Your specific suggestions here] Confidence: [score]

Attribution: [Your attribution here] Confidence: [score]

Attribution tag: [Your tag here]

Issue 2:
...
"""

        total_tokens = self.vector_search.count_tokens(prompt) + 500
        if total_tokens > 4097:
            # Truncate the analysis further or summarize
            aggregated_analysis = {
                "Thoughts": self.truncate_text(aggregated_analysis['Thoughts'], 300),
                "Fetched Rules": self.truncate_text(aggregated_analysis['Fetched Rules'], 300),
                "Fetched Data": self.truncate_text(aggregated_analysis['Fetched Data'], 300)
            }
            prompt = f"""
You are an expert in payment systems analysis. Based on the aggregated analysis below, provide a comprehensive Root Cause Analysis with confidence scores.

**Order ID:** {self.udf_order_id}

**Thought Process:**
{aggregated_analysis['Thoughts']}

**Fetched Rules:**
{aggregated_analysis['Fetched Rules']}

**Fetched Data:**
{aggregated_analysis['Fetched Data']}

**Instructions:**
1. **Root Cause Analysis:** Identify all root causes by referencing specific rules and data points. Explain how each referenced item contributes to the conclusions. Provide confidence scores between 0 and 1 indicating your certainty for each identified root cause.

2. **Suggestions:** Provide actionable steps to resolve each identified issue. Include confidence scores for each suggestion.

3. **Attribution:** Determine whether each issue stems from an external factor (Merchant or Gateway), an internal factor (Juspay), or if it's ambiguous based on the available data. Provide confidence scores for each attribution.

4. **Attribution Tags:** Provide one of the following tags for each issue: MERCHANT | GATEWAY | JUSPAY | AMBIGUOUS.

**Please follow the above structure and level of detail in your response, including confidence scores for each section.**

**Format your response as follows:**

Issue 1:
Root Cause Analysis: [Your detailed analysis here] Confidence: [score]

Suggestion: [Your specific suggestions here] Confidence: [score]

Attribution: [Your attribution here] Confidence: [score]

Attribution tag: [Your tag here]

Issue 2:
...
"""

        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0,
            )
            rca = response.choices[0].text.strip()
            logging.info(f"Aggregated RCA generated with confidence scores: {rca}")
            return rca
        except openai.error.InvalidRequestError as e:
            logging.error(f"InvalidRequestError: {e}")
            # Implement fallback
            return "Aggregated RCA generation failed due to excessive prompt length. Please review the analysis steps."
        except Exception as e:
            logging.error(f"Error generating aggregated RCA: {e}")
            return ""

    def collect_analysis(self, node, max_thoughts=10, max_rules=10, max_data=10):
        thoughts = []
        fetched_rules = []
        fetched_data = []
        current = node
        count = 0
        while current and count < max_thoughts:
            thoughts.append(f"- {current.description}")
            if current.fetched_rules:
                for rule_id, rule_content in current.fetched_rules.items():
                    fetched_rules.append(f"Rule {rule_id}: {rule_content}")
                    if len(fetched_rules) >= max_rules:
                        break
            if current.fetched_data:
                for field, value in current.fetched_data.items():
                    fetched_data.append(f"{field}: {value}")
                    if len(fetched_data) >= max_data:
                        break
            current = current.parent
            count += 1
        analysis = {
            "Thoughts": '\n'.join(reversed(thoughts)),
            "Fetched Rules": '\n'.join(reversed(fetched_rules)),
            "Fetched Data": '\n'.join(reversed(fetched_data))
        }
        return analysis
    
    def visualize_graph(self, final=False):
        plt.figure(figsize=(20, 15))

        if self.pos is None or len(self.pos) != len(self.graph.nodes):
            self.pos = nx.spring_layout(self.graph, k=0.5, iterations=50)

        # Ensure all nodes have positions
        for node in self.graph.nodes:
            if node not in self.pos:
                # If a node doesn't have a position, assign it a random one
                self.pos[node] = np.random.rand(2)

        node_colors = []
        for node in self.graph.nodes:
            if node == self.root.description:
                node_colors.append('lightblue')
            elif node in self.traversed_path:
                node_colors.append('yellow')
            else:
                node_colors.append('lightgreen')

        if final and node_colors:
            node_colors[-1] = 'lightcoral'

        nx.draw(self.graph, self.pos, node_color=node_colors, edge_color='lightgray', with_labels=False,
                node_size=3000, node_shape="o", alpha=0.7, linewidths=1,
                font_size=8, font_weight="bold", arrows=True)

        path_edges = list(zip(self.traversed_path[:-1], self.traversed_path[1:]))
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=path_edges, edge_color='red', width=2)

        pos_attrs = {}
        for node, coords in self.pos.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)

        node_labels = nx.get_node_attributes(self.graph, 'node')
        custom_labels = {node: f"{data.description[:50]}..." if len(data.description) > 50 else data.description
                         for node, data in node_labels.items()}
        nx.draw_networkx_labels(self.graph, pos_attrs, labels=custom_labels, font_size=6)

        if final:
            plt.title(f"Final Thought Graph with Traversed Path - Order ID: {self.udf_order_id}", fontsize=20)
        else:
            plt.title(f"Current Thought Graph with Traversed Path - Order ID: {self.udf_order_id}", fontsize=20)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Root Node', markerfacecolor='lightblue', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Traversed Node', markerfacecolor='yellow', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Unvisited Node', markerfacecolor='lightgreen', markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Final Node', markerfacecolor='lightcoral', markersize=15),
            plt.Line2D([0], [0], color='red', lw=2, label='Traversed Path')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        filename = f"thought_graph_{self.udf_order_id}_{'final' if final else 'current'}.png"
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Graph visualization with traversed path generated and saved as {filename}")

        
    def generate_rca(self, node):
        logging.info(f"Generating RCA for node: {node.description}")

        # Collect all available data with limits
        available_data = self.collect_analysis(node, max_thoughts=10, max_rules=10, max_data=10)

        prompt = f"""
You are an expert in payment systems analysis. Based on the analysis below, provide a detailed Root Cause Analysis with confidence scores.

**Order ID:** {self.udf_order_id}

**Thought Process:**
{available_data['Thoughts']}

**Fetched Rules:**
{available_data['Fetched Rules']}

**Fetched Data:**
{available_data['Fetched Data']}

**Instructions:**
1. **Root Cause Analysis:** Identify the root cause by referencing specific rules and data points. Explain how each referenced item contributes to the conclusion. Provide a confidence score between 0 and 1 indicating your certainty.

2. **Suggestion:** Provide actionable steps to resolve the issue, based on the identified root cause. Include confidence scores for each suggestion.

3. **Attribution:** Determine whether the issue stems from an external factor (Merchant or Gateway), an internal factor (Juspay), or if it's ambiguous based on the available data. Provide confidence scores for each attribution.

4. **Attribution Tag:** Provide one of the following tags: MERCHANT | GATEWAY | JUSPAY | AMBIGUOUS.

**Please follow the above structure and level of detail in your response, including confidence scores for each section.**

**Format your response as follows:**

Root Cause Analysis: [Your detailed analysis here] Confidence: [score]

Suggestion: [Your specific suggestions here] Confidence: [score]

Attribution: [Your attribution here] Confidence: [score]

Attribution tag: [Your tag here]
"""

        total_tokens = self.vector_search.count_tokens(prompt) + 500
        if total_tokens > 4097:
            # Truncate the analysis further or summarize
            available_data = self.collect_analysis(node, max_thoughts=5, max_rules=5, max_data=5)
            prompt = f"""
You are an expert in payment systems analysis. Based on the analysis below, provide a detailed Root Cause Analysis with confidence scores.

**Order ID:** {self.udf_order_id}

**Thought Process:**
{available_data['Thoughts']}

**Fetched Rules:**
{available_data['Fetched Rules']}

**Fetched Data:**
{available_data['Fetched Data']}

**Instructions:**
1. **Root Cause Analysis:** Identify the root cause by referencing specific rules and data points. Explain how each referenced item contributes to the conclusion. Provide a confidence score between 0 and 1 indicating your certainty.

2. **Suggestion:** Provide actionable steps to resolve the issue, based on the identified root cause. Include confidence scores for each suggestion.

3. **Attribution:** Determine whether the issue stems from an external factor (Merchant or Gateway), an internal factor (Juspay), or if it's ambiguous based on the available data. Provide confidence scores for each attribution.

4. **Attribution Tag:** Provide one of the following tags: MERCHANT | GATEWAY | JUSPAY | AMBIGUOUS.

**Please follow the above structure and level of detail in your response, including confidence scores for each section.**

**Format your response as follows:**

Root Cause Analysis: [Your detailed analysis here] Confidence: [score]

Suggestion: [Your specific suggestions here] Confidence: [score]

Attribution: [Your attribution here] Confidence: [score]

Attribution tag: [Your tag here]
"""

        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0,
            )
            rca = response.choices[0].text.strip()
            logging.info(f"RCA generated with confidence scores: {rca}")
            return rca
        except openai.error.InvalidRequestError as e:
            logging.error(f"InvalidRequestError: {e}")
            # Implement fallback
            return "RCA generation failed due to excessive prompt length. Please review the analysis steps."
        except Exception as e:
            logging.error(f"Error generating RCA: {e}")
            return ""

    def force_generate_rca_based_on_all(self):
        logging.info("Force generating RCA based on all analysis steps.")
        # Aggregate all analysis steps
        aggregated_analysis = {
            "Thoughts": set(),
            "Fetched Rules": set(),
            "Fetched Data": set()
        }
        for step in self.analysis_steps:
            aggregated_analysis["Thoughts"].add(f"- {step['Thought']}")
            for rule_id, rule_content in step['Fetched Rules'].items():
                aggregated_analysis["Fetched Rules"].add(f"Rule {rule_id}: {rule_content}")
            for field, value in step['Fetched Data'].items():
                aggregated_analysis["Fetched Data"].add(f"{field}: {value}")

        # Convert sets to sorted strings
        aggregated_analysis = {k: '\n'.join(sorted(v)) for k, v in aggregated_analysis.items()}

        prompt = f"""
You are an expert in payment systems analysis. Based on the aggregated analysis below, provide a comprehensive Root Cause Analysis even if the information is incomplete.

**Order ID:** {self.udf_order_id}

**Analysis so far:**
{aggregated_analysis['Thoughts']}

**Fetched Rules:**
{aggregated_analysis['Fetched Rules']}

**Fetched Data:**
{aggregated_analysis['Fetched Data']}

Please analyze step-by-step and provide an initial root cause response in the following format:

Issue 1:
Root Cause Analysis: [Detailed analysis here] Confidence: [score]

Suggestion: [Specific suggestions here] Confidence: [score]

Attribution: [Attribution here] Confidence: [score]

Attribution tag: [Tag here]

Issue 2:
...
"""

        # Calculate total tokens
        total_tokens = self.vector_search.count_tokens(prompt) + 300
        if total_tokens > 4097:

            aggregated_analysis = {
                "Thoughts": self.truncate_text(aggregated_analysis['Thoughts'], 200),
                "Fetched Rules": self.truncate_text(aggregated_analysis['Fetched Rules'], 200),
                "Fetched Data": self.truncate_text(aggregated_analysis['Fetched Data'], 200)
            }
            prompt = f"""
You are an expert in payment systems analysis. Based on the aggregated analysis below, provide a comprehensive Root Cause Analysis even if the information is incomplete.

**Order ID:** {self.udf_order_id}

**Analysis so far:**
{aggregated_analysis['Thoughts']}

**Fetched Rules:**
{aggregated_analysis['Fetched Rules']}

**Fetched Data:**
{aggregated_analysis['Fetched Data']}

Please analyze step-by-step and provide an initial root cause response in the following format:

Issue 1:
Root Cause Analysis: [Detailed analysis here] Confidence: [score]

Suggestion: [Specific suggestions here] Confidence: [score]

Attribution: [Attribution here] Confidence: [score]

Attribution tag: [Tag here]

Issue 2:
...
"""

        try:
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=300,
                temperature=0,
            )
            rca = response.choices[0].text.strip()
            logging.info(f"Forced RCA generated: {rca}")
            return rca
        except openai.error.InvalidRequestError as e:
            logging.error(f"InvalidRequestError: {e}")
            # Implement fallback
            return "RCA generation failed due to excessive prompt length. Please review the analysis steps."
        except Exception as e:
            logging.error(f"Error forcing RCA generation: {e}")
            return ""

    def generate_summary(self, rca, forced=False):
        """
        Generates a detailed summary of how RCA was reached.
        """
        summary = f"Root Cause Analysis Summary for Order ID: {self.udf_order_id}\n"
        summary += "="*50 + "\n\n"

        summary += "Traversal Path:\n"
        for step in self.traversed_path:
            summary += f"- {step}\n"
        summary += "\n"

        summary += "Analysis Steps:\n"
        for idx, step in enumerate(self.analysis_steps, 1):
            summary += f"Step {idx}:\n"
            summary += f"Thought: {step['Thought']}\n"
            summary += f"Fetched Rules: {json.dumps(step['Fetched Rules'], indent=2)}\n"
            summary += f"Fetched Data: {json.dumps(step['Fetched Data'], indent=2)}\n\n"

        summary += "Final Root Cause Analysis:\n"
        summary += f"{rca}\n\n"

        if forced:
            summary += "Note: RCA was generated despite not meeting the confidence threshold.\n"

        # Save summary to a file
        report_filename = f"RCA_Report_{self.udf_order_id}.txt"
        try:
            with open(report_filename, 'w') as f:
                f.write(summary)
            logging.info(f"RCA summary report generated and saved as {report_filename}")
            print(f"\nDetailed RCA Summary has been saved to {report_filename}")
        except Exception as e:
            logging.error(f"Error saving RCA summary report: {e}")
            print("Failed to save the RCA summary report.")

    def update_target_network(self):
        """
        Updates the target Q-values from the main Q-values to stabilize training.
        """
        logging.info("Updating target network Q-values.")
        for node in self.graph.nodes:
            node_obj = self.graph.nodes[node]['node']
            node_obj.target_q_value = node_obj.q_value
        logging.info("Target network Q-values updated successfully.")

    def run_analysis(self):
        logging.info("Starting analysis")
        self.root.heuristic = self.calculate_heuristic(self.root)
        priority = self.compute_priority(self.root)
        self.queue.put((priority, self.root))

        best_q_value = float('-inf')
        self.iterations_without_improvement = 0

        start_time = time.time()
        min_time = 30 
        max_time = 120 

        while not self.queue.empty() and self.iterations < self.max_iterations:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > max_time:
                logging.info("Maximum time limit reached; terminating analysis.")
                break

            node = self.select_next_node()
            if node is None:
                logging.warning("No node selected; terminating analysis.")
                break

            self.expand_node(node)
            self.iterations += 1
            logging.info(f"Iteration {self.iterations}: Expanded node '{node.description}' with Q-value {node.q_value}")

            reward = self.calculate_reward(node)
            self.update_q_value(node, reward)
            logging.info(f"Updated Q-value for node '{node.description}': {node.q_value}")

            # Track best Q-value for convergence
            if node.q_value > best_q_value:
                best_q_value = node.q_value
                self.iterations_without_improvement = 0
                logging.info(f"New best Q-value found: {best_q_value}")
            else:
                self.iterations_without_improvement += 1
                logging.info(f"No improvement in Q-value. Iterations without improvement: {self.iterations_without_improvement}")

            # Adjust parameters adaptively
            self.adjust_epsilon(self.iterations_without_improvement)

            # Visualize graph at intervals
            if self.iterations % self.visualization_interval == 0:
                self.visualize_graph()

            # Ensure minimum run time
            if elapsed_time < min_time:
                continue

            # Early convergence conditions
            if len(self.rca_findings) >= 5 and elapsed_time >= min_time:
                logging.info("Multiple high-confidence RCA findings obtained; terminating analysis early.")
                break

            if self.iterations_without_improvement > self.iteration_limit_without_improvement and elapsed_time >= min_time:
                logging.info(f"No improvement over {self.iteration_limit_without_improvement} iterations; terminating analysis.")
                break

            # Log performance metrics
            logging.info(f"Iteration {self.iterations}: Elapsed Time: {elapsed_time:.2f}s, Epsilon: {self.epsilon:.4f}, Alpha: {self.alpha:.4f}, Gamma: {self.gamma:.4f}")

            # Update target network if frequency is met
            self.iterations_since_target_update += 1
            if self.iterations_since_target_update >= self.target_q_update_frequency:
                self.update_target_network()
                self.iterations_since_target_update = 0

        # After search, aggregate all RCA findings
        if self.rca_findings:
            rca = self.generate_aggregated_rca()
            if rca:
                print("Final Root Cause Analysis:")
                print(rca)
                self.generate_summary(rca)
            else:
                print("Unable to determine a definitive Root Cause Analysis with the available information.")
        else:
            # Attempt to force RCA generation if no findings
            rca = self.force_generate_rca_based_on_all()
            if rca:
                print("Final Root Cause Analysis:")
                print(rca)
                self.generate_summary(rca, forced=True)
            else:
                print("Unable to determine a definitive Root Cause Analysis with the available information.")

        # Visualize final graph
        self.visualize_graph(final=True)

        # Log the final traversal path
        logging.info(f"Final Traversal Path: {self.traversed_path}")

# def main():
def analyze(udf_order_id, udf_merchant_id, log, merchant_details, transaction_details):

    data_loader = DataLoader(log, merchant_details, transaction_details)

    # Convert context to JSON rules
    rules_context = data_loader.convert_context_to_json_rules()
    if not data_loader.rules_json:
        logging.error("Rules JSON is empty. Exiting.")
        return

    # Load fields context
    fields_context = data_loader.fields_context

    vector_search = VectorSearch()
    vector_search.load_or_build_vectors(rules_context, fields_context)

    logging.info("All embeddings and context files have been generated and saved successfully.")

    # Initialize ExecutorModuleRules with reference to DecisionModuleRules for dynamic updates
    decision_module_rules = DecisionModuleRules(data_loader.rules_context, vector_search)
    executor_module_rules = ExecutorModuleRules(data_loader.rules_context, decision_module_rules)

    executor_module_json = ExecutorModuleJSON(data_loader)

    config = {
        'iteration_limit': 1000,
        'iteration_limit_without_improvement': 100,
        'max_iterations': 800,
        'max_depth': 25,
        'recurring_threshold': 3,
        'min_exploration_depth': 3,
        'rca_confidence_threshold': 0.7,
        'convergence_threshold': 0.005,
        'breadth': 7,
        'visualization_interval': 15,
        'replay_buffer_size': 100,
        'exploration_constant': 1.9,  # UCB exploration parameter
        'target_q_update_frequency': 100  # Frequency to update target network
    }

    got_manager = GraphOfThoughts(udf_order_id, data_loader, vector_search, config=config)
    got_manager.run_analysis()

