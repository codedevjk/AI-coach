import os
import gradio as gr
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import librosa
import time
import warnings

# --- Configuration for Cache Directories (ADD THIS BLOCK FIRST) ---
# Set cache directories to a writable location within the container
CACHE_DIR = "/app/cache" # Or "/tmp/cache"
os.makedirs(CACHE_DIR, exist_ok=True) # Ensure the directory exists
os.environ["HF_HOME"] = CACHE_DIR # Hugging Face cache
os.environ["TORCH_HOME"] = CACHE_DIR # Torch cache (might be used by some models)
# Whisper uses a specific download function, we'll handle its cache in the load call

# --- Configuration ---
# Whisper model name should be the local name (e.g., 'tiny', 'base')
# 'tiny' is multilingual, 'tiny.en' is English-only but faster/ more accurate for English
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "google/gemma-2b-it") # Hugging Face model ID
HF_TOKEN = os.getenv("HF_TOKEN") # Get token from environment variable
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Using cache directory: {CACHE_DIR}")

#--- Load Models (on startup) ---
print("Loading Whisper model...")
# Whisper expects the local model name directly
# Pass the download_root argument to specify the cache location for Whisper models
whisper_model = whisper.load_model(WHISPER_MODEL_NAME, download_root=os.path.join(CACHE_DIR, "whisper"))

# --- Load LLM Model (Robust CPU/GPU handling) ---
print("Loading LLM tokenizer and model...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, token=HF_TOKEN, cache_dir=CACHE_DIR)

# Load model with explicit settings based on device
if DEVICE == "cpu":
    print("Loading LLM for CPU with low_memory usage settings...")
    # Critical settings for CPU to avoid offload errors and reduce memory
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.float32, # Use float32 for CPU stability
        low_cpu_mem_usage=True,    # Crucial for loading on CPU with limited RAM
        cache_dir=CACHE_DIR,
        # DO NOT use device_map="auto" on CPU here, let PyTorch handle it
    )
    # Explicitly move the model to CPU (safeguard)
    llm_model = llm_model.to(DEVICE)
else: # DEVICE == "cuda"
    print("Loading LLM for GPU...")
    # Settings optimized for GPU
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16, # bfloat16 is efficient on modern GPUs
        device_map="auto",          # Allow accelerate to manage GPU distribution
        cache_dir=CACHE_DIR
    )
# Ensure the model is in evaluation mode
llm_model.eval()
print("LLM model loaded and set to eval mode.")

# --- Helper Functions ---

def get_interview_questions(role):
    """Provides a list of questions based on the selected job role."""
    questions = {
    "react": [
        "What is React and what are its key features?",
        "Explain the difference between functional and class components.",
        "What is JSX and how does it work?",
        "What are React Hooks? Explain useState and useEffect.",
        "What is the Virtual DOM and how does it work?",
        "Explain the component lifecycle methods.",
        "What is state management and how do you handle it in React?",
        "What are controlled vs uncontrolled components?",
        "Explain React Router and its implementation.",
        "What is Context API and when would you use it?",
        "How do you optimize React application performance?",
        "What are Higher Order Components (HOCs)?",
        "Explain React.memo and useMemo.",
        "What is the difference between props and state?",
        "How do you handle forms in React?"
    ],
    
    "javascript": [
        "What are the different data types in JavaScript?",
        "Explain hoisting in JavaScript.",
        "What is the difference between var, let, and const?",
        "What are closures and how do they work?",
        "Explain the concept of prototypes and inheritance.",
        "What is the event loop in JavaScript?",
        "What are promises and how do they work?",
        "Explain async/await vs promises.",
        "What is the difference between == and === operators?",
        "What are arrow functions and how do they differ from regular functions?",
        "Explain call, apply, and bind methods.",
        "What is destructuring in JavaScript?",
        "What are template literals and their benefits?",
        "Explain the concept of currying.",
        "What is the difference between synchronous and asynchronous programming?"
    ],
    
    "java": [
        "Explain the differences between JDK, JRE, and JVM.",
        "What are the main principles of OOPs (Object-Oriented Programming) in Java?",
        "What is the difference between `==` and `.equals()` method in Java?",
        "Explain the concept of inheritance in Java.",
        "What is method overloading and method overriding in Java?",
        "What is a constructor in Java? What are the different types?",
        "Explain the `final` keyword in Java.",
        "What is the difference between an abstract class and an interface in Java?",
        "What are collections in Java? Explain the Collection framework.",
        "What is the difference between `ArrayList` and `LinkedList`?",
        "Explain the `HashMap` internal implementation.",
        "What is exception handling in Java? Explain `try`, `catch`, `finally`.",
        "What is multithreading in Java? How can you create a thread?",
        "What is the difference between `String`, `StringBuilder`, and `StringBuffer`?",
        "What are the access modifiers in Java? Explain `public`, `private`, `protected`, and default."
    ],
    
    "python": [
        "What are the key features of Python?",
        "Explain the difference between lists, tuples, and dictionaries.",
        "What are decorators in Python?",
        "Explain the concept of generators and iterators.",
        "What is the difference between deep copy and shallow copy?",
        "What are lambda functions and when to use them?",
        "Explain exception handling in Python (try, except, finally).",
        "What is the Global Interpreter Lock (GIL)?",
        "What are Python modules and packages?",
        "Explain list comprehensions and their benefits.",
        "What is the difference between *args and **kwargs?",
        "What are class methods, static methods, and instance methods?",
        "Explain the concept of multiple inheritance in Python.",
        "What is the difference between Python 2 and Python 3?",
        "What are context managers and the 'with' statement?"
    ],
    
    "sql": [
        "What is the difference between SQL and NoSQL databases?",
        "Explain the different types of SQL joins.",
        "What is normalization and its different forms?",
        "What are indexes and how do they improve performance?",
        "Explain the difference between UNION and UNION ALL.",
        "What are stored procedures and functions?",
        "What is the difference between DELETE, DROP, and TRUNCATE?",
        "Explain ACID properties in databases.",
        "What are triggers and when are they used?",
        "What is the difference between clustered and non-clustered indexes?",
        "Explain the concept of database transactions.",
        "What are views and their advantages?",
        "What is the difference between primary key and unique key?",
        "Explain subqueries and their types.",
        "What are window functions and their use cases?"
    ],
    
    "angular": [
        "What is Angular and how does it differ from AngularJS?",
        "Explain the Angular architecture and its building blocks.",
        "What are components, services, and modules in Angular?",
        "What is dependency injection in Angular?",
        "Explain data binding in Angular (one-way, two-way).",
        "What are directives and their types?",
        "What is the Angular CLI and its benefits?",
        "Explain routing in Angular applications.",
        "What are observables and how are they used in Angular?",
        "What is the difference between ViewChild and ContentChild?",
        "Explain Angular lifecycle hooks.",
        "What are pipes in Angular and how to create custom pipes?",
        "What is lazy loading and how to implement it?",
        "Explain Angular forms (template-driven vs reactive).",
        "What are guards in Angular routing?"
    ],
    
    "node.js": [
        "What is Node.js and how does it work?",
        "Explain the event loop in Node.js.",
        "What is the difference between synchronous and asynchronous operations?",
        "What are callbacks, promises, and async/await in Node.js?",
        "What is NPM and package.json?",
        "Explain middleware in Express.js.",
        "What is the difference between require() and import?",
        "How do you handle file operations in Node.js?",
        "What are streams in Node.js?",
        "Explain error handling in Node.js.",
        "What is clustering in Node.js?",
        "What are the core modules in Node.js?",
        "How do you connect to databases in Node.js?",
        "What is the difference between process.nextTick() and setImmediate()?",
        "Explain RESTful API development with Node.js."
    ],
    
    "aws": [
        "What is cloud computing and its service models (IaaS, PaaS, SaaS)?",
        "What are the core services of AWS?",
        "Explain EC2 and its instance types.",
        "What is the difference between S3 and EBS?",
        "What is AWS Lambda and serverless computing?",
        "Explain VPC and its components.",
        "What is the difference between horizontal and vertical scaling?",
        "What are security groups and NACLs?",
        "Explain load balancing in AWS.",
        "What is CloudFormation and Infrastructure as Code?",
        "What is the difference between RDS and DynamoDB?",
        "Explain AWS IAM and its components.",
        "What is CloudWatch and monitoring in AWS?",
        "What are availability zones and regions?",
        "Explain disaster recovery strategies in AWS."
    ],
    
    "docker": [
        "What is Docker and containerization?",
        "What is the difference between containers and virtual machines?",
        "Explain Docker architecture and its components.",
        "What is a Dockerfile and how to write one?",
        "What are Docker images and containers?",
        "Explain Docker networking modes.",
        "What are Docker volumes and their types?",
        "What is the difference between ADD and COPY in Dockerfile?",
        "How do you optimize Docker images?",
        "What is Docker Compose and its use cases?",
        "Explain multi-stage builds in Docker.",
        "What are Docker registries and repositories?",
        "How do you handle data persistence in Docker?",
        "What is the difference between RUN, CMD, and ENTRYPOINT?",
        "Explain Docker security best practices."
    ],
    
    "kubernetes": [
        "What is Kubernetes and its architecture?",
        "Explain pods, nodes, and clusters in Kubernetes.",
        "What are services and their types in Kubernetes?",
        "What is the difference between Deployment and StatefulSet?",
        "Explain ConfigMaps and Secrets in Kubernetes.",
        "What are namespaces and their purpose?",
        "What is kubectl and common commands?",
        "Explain horizontal pod autoscaling.",
        "What are ingress controllers and their use?",
        "What is the difference between ReplicaSet and Deployment?",
        "Explain persistent volumes and persistent volume claims.",
        "What are DaemonSets and when to use them?",
        "What is the Kubernetes control plane?",
        "Explain rolling updates and rollbacks.",
        "What are resource quotas and limits?"
    ],
    
    "git": [
        "What is Git and how does it differ from other VCS?",
        "Explain the Git workflow and basic commands.",
        "What is the difference between git merge and git rebase?",
        "What are branches and how to manage them?",
        "Explain git stash and its use cases.",
        "What is the difference between git pull and git fetch?",
        "How do you resolve merge conflicts?",
        "What are git hooks and their types?",
        "Explain the difference between HEAD, working tree, and index.",
        "What is git cherry-pick and when to use it?",
        "How do you undo changes in Git?",
        "What are remote repositories and how to manage them?",
        "Explain git submodules and their use.",
        "What is the difference between git reset and git revert?",
        "How do you configure Git for team collaboration?"
    ],
    
    "machine learning": [
        "Describe a machine learning project you've worked on.",
        "How do you handle overfitting in your models?",
        "Explain the Bias-Variance Tradeoff.",
        "What is the difference between supervised, unsupervised, and reinforcement learning?",
        "Describe the steps involved in building a machine learning model.",
        "What is cross-validation, and why is it important?",
        "Explain the concept of regularization. When and why do we use it?",
        "What is the difference between bagging and boosting?",
        "How do you handle missing data in a dataset?",
        "What is the ROC curve and AUC (Area Under the Curve)?",
        "Explain the difference between Type I and Type II errors.",
        "What is the vanishing gradient problem? How can it be addressed?",
        "What are decision trees and random forests?",
        "Explain linear regression and logistic regression.",
        "What are evaluation metrics for ML models?"
    ],
    
    "data science": [
        "How do you approach cleaning a messy dataset?",
        "Tell me about a time you discovered an interesting insight in data.",
        "What tools do you use for data visualization?",
        "What is the difference between data mining and data analysis?",
        "How do you handle missing or corrupted data in a dataset?",
        "What are the steps involved in conducting a data analysis project?",
        "Explain the concept of outliers. How would you detect and handle them?",
        "What is data cleaning and wrangling? Why is it important?",
        "Differentiate between univariate, bivariate, and multivariate analysis.",
        "What is the difference between descriptive, inferential, and predictive analytics?",
        "Explain the concept of correlation and covariance.",
        "What are the different types of sampling techniques you know?",
        "What is A/B testing, and how is it used in data analysis?",
        "How do you assess the quality of a dataset?",
        "What are some common data visualization techniques, and when would you use them?"
    ],
    
    "devops": [
        "What is DevOps and its core principles?",
        "Explain the DevOps lifecycle and methodologies.",
        "What is CI/CD and its benefits?",
        "What is Infrastructure as Code (IaC)?",
        "Explain monitoring and logging in DevOps.",
        "What are containers and their role in DevOps?",
        "What is configuration management?",
        "Explain the concept of shift-left testing.",
        "What are blue-green and canary deployments?",
        "What is the difference between DevOps and traditional development?",
        "Explain microservices architecture in DevOps context.",
        "What are the key DevOps tools and their categories?",
        "What is site reliability engineering (SRE)?",
        "Explain security in DevOps (DevSecOps).",
        "What are the cultural aspects of DevOps adoption?"
    ],
    
    "microservices": [
        "What are microservices and their characteristics?",
        "What is the difference between monolithic and microservices architecture?",
        "What are the benefits and challenges of microservices?",
        "How do microservices communicate with each other?",
        "What is service discovery and its mechanisms?",
        "Explain API Gateway and its role.",
        "What are circuit breakers and their purpose?",
        "How do you handle data management in microservices?",
        "What is distributed tracing and monitoring?",
        "Explain the saga pattern for distributed transactions.",
        "What are the design patterns for microservices?",
        "How do you handle security in microservices?",
        "What is container orchestration for microservices?",
        "Explain event-driven architecture in microservices.",
        "What are the best practices for microservices deployment?"
    ],
    
    "rest api": [
        "What is REST and its architectural principles?",
        "What are HTTP methods and their usage in REST?",
        "What is the difference between REST and SOAP?",
        "Explain HTTP status codes and their meanings.",
        "What is JSON and why is it preferred in REST APIs?",
        "How do you handle authentication and authorization in REST APIs?",
        "What is the difference between PUT and PATCH methods?",
        "Explain API versioning strategies.",
        "What are HATEOAS and its implementation?",
        "How do you handle pagination in REST APIs?",
        "What is rate limiting and throttling?",
        "Explain caching strategies for REST APIs.",
        "What are the best practices for REST API design?",
        "How do you handle errors in REST APIs?",
        "What is OpenAPI/Swagger specification?"
    ],
    
    "cloud computing": [
        "What is cloud computing and its characteristics?",
        "Explain the cloud service models (IaaS, PaaS, SaaS).",
        "What are the cloud deployment models?",
        "What are the benefits and challenges of cloud computing?",
        "Explain virtualization and its role in cloud computing.",
        "What is multi-tenancy in cloud environments?",
        "What are the key considerations for cloud migration?",
        "Explain cloud security and compliance.",
        "What is serverless computing and its benefits?",
        "What are the different cloud pricing models?",
        "Explain disaster recovery in cloud environments.",
        "What is hybrid cloud and its use cases?",
        "What are containers and their role in cloud computing?",
        "Explain cloud monitoring and management.",
        "What are the major cloud service providers and their offerings?"
    ],
    
    "spring boot": [
        "What is Spring Boot and its advantages?",
        "Explain the Spring Boot architecture.",
        "What are Spring Boot starters and their purpose?",
        "What is auto-configuration in Spring Boot?",
        "Explain dependency injection in Spring Boot.",
        "What are Spring Boot annotations and their usage?",
        "How do you create REST APIs with Spring Boot?",
        "What is the difference between @Component, @Service, and @Repository?",
        "Explain Spring Boot profiles and configuration.",
        "What is Spring Boot Actuator and its features?",
        "How do you handle exceptions in Spring Boot?",
        "What is Spring Data JPA and its benefits?",
        "Explain security implementation in Spring Boot.",
        "What are Spring Boot testing strategies?",
        "How do you deploy Spring Boot applications?"
    ],
    
    "linux": [
        "What is Linux and its key features?",
        "Explain the Linux file system hierarchy.",
        "What are the basic Linux commands for file operations?",
        "What is the difference between absolute and relative paths?",
        "Explain file permissions and ownership in Linux.",
        "What are processes and how to manage them?",
        "What is the difference between hard links and soft links?",
        "Explain shell scripting and its basics.",
        "What are environment variables and their usage?",
        "How do you manage services in Linux?",
        "What is cron and how to schedule tasks?",
        "Explain network configuration in Linux.",
        "What are package managers and their usage?",
        "How do you monitor system performance in Linux?",
        "What are log files and their importance?"
    ],
    
    "agile": [
        "What is Agile methodology and its principles?",
        "Explain the difference between Agile and Waterfall.",
        "What are the popular Agile frameworks (Scrum, Kanban)?",
        "What are user stories and their components?",
        "Explain the Scrum process and its roles.",
        "What are sprints and sprint planning?",
        "What is the difference between epic, story, and task?",
        "Explain daily standups and their purpose.",
        "What are retrospectives and their importance?",
        "What is the definition of done (DoD)?",
        "How do you handle changing requirements in Agile?",
        "What are burn-down and burn-up charts?",
        "Explain the concept of velocity in Scrum.",
        "What is the difference between Scrum Master and Product Owner?",
        "How do you estimate user stories in Agile?"
    ],
    
    "typescript": [
        "What is TypeScript and its advantages over JavaScript?",
        "Explain the basic types in TypeScript.",
        "What are interfaces in TypeScript?",
        "What is the difference between interface and type in TypeScript?",
        "Explain generics in TypeScript.",
        "What are union and intersection types?",
        "What is type assertion in TypeScript?",
        "Explain decorators in TypeScript.",
        "What are modules and namespaces in TypeScript?",
        "How do you handle null and undefined in TypeScript?",
        "What is the any type and when should you avoid it?",
        "Explain enums in TypeScript.",
        "What are utility types in TypeScript?",
        "How do you configure TypeScript compiler?",
        "What is the difference between TypeScript and JavaScript compilation?"
    ],
    
    "c#": [
        "What is C# and its key features?",
        "Explain the .NET framework and .NET Core.",
        "What are the basic data types in C#?",
        "What is the difference between value and reference types?",
        "Explain object-oriented programming concepts in C#.",
        "What are access modifiers in C#?",
        "What is the difference between abstract class and interface in C#?",
        "Explain inheritance and polymorphism in C#.",
        "What are generics in C#?",
        "What is LINQ and how is it used?",
        "Explain exception handling in C#.",
        "What are delegates and events in C#?",
        "What is the difference between String and StringBuilder?",
        "Explain garbage collection in C#.",
        "What are nullable types in C#?"
    ],
    
    "asp.net": [
        "What is ASP.NET and its architecture?",
        "Explain the difference between ASP.NET Web Forms and ASP.NET MVC.",
        "What is the MVC pattern and how is it implemented in ASP.NET?",
        "What are controllers, models, and views in ASP.NET MVC?",
        "Explain the ASP.NET page lifecycle.",
        "What is dependency injection in ASP.NET Core?",
        "How do you handle routing in ASP.NET MVC?",
        "What are action filters in ASP.NET MVC?",
        "Explain authentication and authorization in ASP.NET.",
        "What is the difference between ViewData, ViewBag, and TempData?",
        "How do you handle exceptions in ASP.NET?",
        "What are areas in ASP.NET MVC?",
        "Explain Web API in ASP.NET.",
        "What is middleware in ASP.NET Core?",
        "How do you implement caching in ASP.NET?"
    ],
    
    "mongodb": [
        "What is MongoDB and its key features?",
        "Explain the difference between SQL and NoSQL databases.",
        "What are documents and collections in MongoDB?",
        "What is BSON and how does it differ from JSON?",
        "Explain the MongoDB architecture.",
        "What are the different data types supported by MongoDB?",
        "How do you perform CRUD operations in MongoDB?",
        "What is indexing in MongoDB and its types?",
        "Explain aggregation framework in MongoDB.",
        "What is sharding in MongoDB?",
        "What are replica sets in MongoDB?",
        "How do you handle relationships in MongoDB?",
        "What is the difference between embedded and referenced documents?",
        "Explain GridFS in MongoDB.",
        "What are the best practices for MongoDB schema design?"
    ],
    
    "postgresql": [
        "What is PostgreSQL and its key features?",
        "Explain the architecture of PostgreSQL.",
        "What are the data types supported by PostgreSQL?",
        "How do you create and manage databases in PostgreSQL?",
        "What are schemas in PostgreSQL?",
        "Explain indexing in PostgreSQL and its types.",
        "What are views and materialized views in PostgreSQL?",
        "How do you handle transactions in PostgreSQL?",
        "What are stored procedures and functions in PostgreSQL?",
        "Explain partitioning in PostgreSQL.",
        "What is VACUUM and its importance in PostgreSQL?",
        "How do you handle JSON data in PostgreSQL?",
        "What are common table expressions (CTEs) in PostgreSQL?",
        "Explain replication in PostgreSQL.",
        "What are the performance tuning techniques for PostgreSQL?"
    ],
    
    "redis": [
        "What is Redis and its key features?",
        "Explain the data structures supported by Redis.",
        "What is the difference between Redis and traditional databases?",
        "How does Redis handle persistence?",
        "What are the different persistence options in Redis?",
        "Explain Redis clustering and its benefits.",
        "What is Redis Sentinel and its purpose?",
        "How do you handle expiration and eviction in Redis?",
        "What are Redis transactions and how do they work?",
        "Explain pub/sub messaging in Redis.",
        "What are Redis modules and their use cases?",
        "How do you monitor Redis performance?",
        "What are the best practices for Redis security?",
        "Explain Redis pipelining and its benefits.",
        "What are the common use cases for Redis?"
    ],
    
    "tensorflow": [
        "What is TensorFlow and its key components?",
        "Explain the difference between TensorFlow 1.x and 2.x.",
        "What are tensors and computational graphs in TensorFlow?",
        "How do you create and train a neural network in TensorFlow?",
        "What is Keras and how does it integrate with TensorFlow?",
        "Explain eager execution in TensorFlow 2.x.",
        "What are TensorFlow estimators and their benefits?",
        "How do you handle data preprocessing in TensorFlow?",
        "What is TensorBoard and its features?",
        "Explain distributed training in TensorFlow.",
        "What are TensorFlow Serving and TensorFlow Lite?",
        "How do you save and load models in TensorFlow?",
        "What are custom layers and losses in TensorFlow?",
        "Explain gradient tape in TensorFlow 2.x.",
        "What are the best practices for TensorFlow model optimization?"
    ],
    
    "pytorch": [
        "What is PyTorch and its key features?",
        "Explain the difference between PyTorch and TensorFlow.",
        "What are tensors and autograd in PyTorch?",
        "How do you create and train neural networks in PyTorch?",
        "What is the difference between torch.nn and torch.nn.functional?",
        "Explain dynamic computational graphs in PyTorch.",
        "What are DataLoaders and Datasets in PyTorch?",
        "How do you handle GPU operations in PyTorch?",
        "What are optimizers and schedulers in PyTorch?",
        "Explain custom loss functions and layers in PyTorch.",
        "What is torchvision and its utilities?",
        "How do you save and load models in PyTorch?",
        "What is distributed training in PyTorch?",
        "Explain PyTorch Lightning and its benefits.",
        "What are the best practices for PyTorch model development?"
    ],
    
    "jenkins": [
        "What is Jenkins and its key features?",
        "Explain the Jenkins architecture and components.",
        "What are Jenkins pipelines and their types?",
        "How do you create a Jenkins job?",
        "What is the difference between declarative and scripted pipelines?",
        "Explain Jenkins plugins and their management.",
        "What are Jenkins agents and nodes?",
        "How do you configure Jenkins for CI/CD?",
        "What is Jenkinsfile and its structure?",
        "How do you handle security in Jenkins?",
        "What are Jenkins environment variables?",
        "Explain parallel execution in Jenkins pipelines.",
        "How do you integrate Jenkins with version control systems?",
        "What are Jenkins build triggers?",
        "What are the best practices for Jenkins pipeline development?"
    ],
    
    "terraform": [
        "What is Terraform and Infrastructure as Code?",
        "Explain the Terraform workflow and lifecycle.",
        "What are Terraform providers and resources?",
        "How do you write Terraform configuration files?",
        "What is Terraform state and its importance?",
        "Explain Terraform modules and their benefits.",
        "What are Terraform variables and outputs?",
        "How do you manage Terraform state remotely?",
        "What are data sources in Terraform?",
        "Explain Terraform workspaces and their use cases.",
        "How do you handle secrets in Terraform?",
        "What is the difference between terraform plan and terraform apply?",
        "How do you structure Terraform projects for large infrastructures?",
        "What are Terraform provisioners and when to use them?",
        "What are the best practices for Terraform development?"
    ],
    
    "react native": [
        "What is React Native and how does it differ from React?",
        "Explain the architecture of React Native applications.",
        "What is the bridge in React Native and how does it work?",
        "How do you handle navigation in React Native?",
        "What are the differences between iOS and Android development in React Native?",
        "Explain state management in React Native applications.",
        "How do you handle platform-specific code in React Native?",
        "What are native modules and how do you create them?",
        "How do you optimize performance in React Native apps?",
        "What is Expo and how does it simplify React Native development?",
        "How do you handle data storage in React Native?",
        "What are the debugging tools available for React Native?",
        "How do you implement push notifications in React Native?",
        "What is the difference between React Native CLI and Expo CLI?",
        "How do you test React Native applications?"
    ],
    
    "flutter": [
        "What is Flutter and what are its key advantages?",
        "Explain the Flutter architecture and its components.",
        "What is Dart and why is it used in Flutter?",
        "What are widgets in Flutter and their types?",
        "Explain the difference between StatefulWidget and StatelessWidget.",
        "What is the widget tree and element tree in Flutter?",
        "How do you handle state management in Flutter?",
        "What are the different state management approaches in Flutter?",
        "How do you handle navigation and routing in Flutter?",
        "What is the difference between hot reload and hot restart?",
        "How do you integrate native code with Flutter?",
        "What are Flutter plugins and packages?",
        "How do you handle responsive design in Flutter?",
        "What are the testing strategies in Flutter?",
        "How do you optimize Flutter app performance?"
    ],
    
    "vue.js": [
        "What is Vue.js and its key features?",
        "Explain the Vue.js lifecycle hooks.",
        "What is the Vue instance and its properties?",
        "What are Vue components and how do you create them?",
        "Explain data binding in Vue.js (one-way and two-way).",
        "What are directives in Vue.js and their types?",
        "What is Vuex and how does state management work?",
        "How do you handle routing in Vue.js applications?",
        "What are computed properties and watchers in Vue.js?",
        "Explain the difference between v-if and v-show.",
        "What are slots in Vue.js and their use cases?",
        "How do you handle forms and validation in Vue.js?",
        "What is the Vue CLI and its features?",
        "Explain the difference between Vue 2 and Vue 3.",
        "How do you optimize Vue.js application performance?"
    ],
    
    "next.js": [
        "What is Next.js and its key features?",
        "Explain the different rendering methods in Next.js (SSR, SSG, CSR).",
        "What is the difference between getServerSideProps and getStaticProps?",
        "How does routing work in Next.js?",
        "What are API routes in Next.js?",
        "Explain dynamic routing in Next.js.",
        "What is Incremental Static Regeneration (ISR)?",
        "How do you handle authentication in Next.js?",
        "What are Next.js middleware and their use cases?",
        "How do you optimize images in Next.js?",
        "What is the difference between pages and app directory in Next.js 13?",
        "How do you handle SEO in Next.js applications?",
        "What are Next.js plugins and how to use them?",
        "How do you deploy Next.js applications?",
        "What are the performance optimization techniques in Next.js?"
    ],
    
    "cybersecurity": [
        "What are the fundamental principles of cybersecurity (CIA Triad)?",
        "Explain the different types of cyber threats and attacks.",
        "What is the difference between vulnerability, threat, and risk?",
        "What are the common authentication and authorization mechanisms?",
        "Explain symmetric vs asymmetric encryption.",
        "What is a firewall and how does it work?",
        "What are the different types of malware?",
        "Explain the concept of penetration testing.",
        "What is the OWASP Top 10 and its significance?",
        "What are security frameworks like NIST and ISO 27001?",
        "How do you secure web applications?",
        "What is incident response and its phases?",
        "Explain network security concepts (VPN, IDS/IPS).",
        "What are security best practices for cloud computing?",
        "How do you conduct risk assessment and management?"
    ],
    
    "graphql": [
        "What is GraphQL and how does it differ from REST?",
        "Explain the core concepts of GraphQL (Schema, Types, Resolvers).",
        "What are queries, mutations, and subscriptions in GraphQL?",
        "How do you define a GraphQL schema?",
        "What are GraphQL resolvers and how do they work?",
        "Explain the N+1 query problem and how to solve it.",
        "What are GraphQL fragments and their benefits?",
        "How do you handle authentication and authorization in GraphQL?",
        "What are the advantages and disadvantages of GraphQL?",
        "How do you implement caching in GraphQL?",
        "What are GraphQL directives and their use cases?",
        "How do you handle file uploads in GraphQL?",
        "What are the popular GraphQL tools and libraries?",
        "How do you test GraphQL APIs?",
        "What are the best practices for GraphQL API design?"
    ],
    
    "go": [
        "What is Go (Golang) and its key features?",
        "Explain the Go workspace and project structure.",
        "What are goroutines and how do they work?",
        "Explain channels in Go and their types.",
        "What is the difference between goroutines and threads?",
        "How does garbage collection work in Go?",
        "What are interfaces in Go and how are they implemented?",
        "Explain pointers and memory management in Go.",
        "What are slices and maps in Go?",
        "How do you handle errors in Go?",
        "What are packages and modules in Go?",
        "Explain the select statement and its use cases.",
        "What are defer statements in Go?",
        "How do you write unit tests in Go?",
        "What are the best practices for Go development?"
    ],
    
    "blockchain": [
        "What is blockchain and how does it work?",
        "Explain the key components of a blockchain (blocks, transactions, hash).",
        "What is the difference between public, private, and consortium blockchains?",
        "What are consensus mechanisms (Proof of Work, Proof of Stake)?",
        "What are smart contracts and how do they work?",
        "Explain cryptocurrency and digital wallets.",
        "What is the difference between Bitcoin and Ethereum?",
        "What are the security features of blockchain?",
        "How do you handle scalability issues in blockchain?",
        "What are the use cases of blockchain beyond cryptocurrency?",
        "What is a fork in blockchain (soft fork vs hard fork)?",
        "Explain decentralized applications (DApps).",
        "What are the challenges and limitations of blockchain?",
        "What is Web3 and its relationship with blockchain?",
        "How do you develop and deploy smart contracts?"
    ],
    
    "apache kafka": [
        "What is Apache Kafka and its key features?",
        "Explain the Kafka architecture (brokers, topics, partitions).",
        "What are producers and consumers in Kafka?",
        "How does Kafka ensure message ordering and delivery?",
        "What are Kafka topics and partitions?",
        "Explain consumer groups and their benefits.",
        "What is Kafka Connect and its use cases?",
        "How do you handle fault tolerance in Kafka?",
        "What are Kafka Streams and their advantages?",
        "How do you monitor Kafka clusters?",
        "What is the difference between Kafka and traditional message queues?",
        "How do you handle schema evolution in Kafka?",
        "What are the best practices for Kafka configuration?",
        "How do you secure Kafka clusters?",
        "What are the common Kafka use cases and patterns?"
    ],
    
    "elasticsearch": [
        "What is Elasticsearch and its key features?",
        "Explain the architecture of Elasticsearch (nodes, indices, shards).",
        "What is the difference between Elasticsearch, Logstash, and Kibana (ELK stack)?",
        "How do you index and search documents in Elasticsearch?",
        "What are mappings and analyzers in Elasticsearch?",
        "Explain different types of queries in Elasticsearch.",
        "How do you handle aggregations in Elasticsearch?",
        "What is the difference between exact match and full-text search?",
        "How do you optimize Elasticsearch performance?",
        "What are Elasticsearch clusters and their management?",
        "How do you handle data ingestion in Elasticsearch?",
        "What are the different node types in Elasticsearch?",
        "How do you backup and restore Elasticsearch data?",
        "What are the security features in Elasticsearch?",
        "What are the common use cases for Elasticsearch?"
    ]
}
    return questions.get(role.lower(), ["Tell me about yourself.", "Why do you want this job?", "What are your strengths and weaknesses?"])

def transcribe_audio(audio_file_path):
    """Transcribes audio using Whisper."""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return ""
    print(f"Transcribing audio: {audio_file_path}")
    # Whisper expects the file path directly
    result = whisper_model.transcribe(audio_file_path)
    transcription = result["text"]
    print(f"Transcription: {transcription}")
    return transcription

def analyze_audio_features(audio_file_path):
    """Analyzes audio for pace, pauses, etc."""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return {"pace": "N/A", "filler_words": 0}
    try:
        # --- Disable Librosa Cache (Fix for Hugging Face Spaces) ---
        # The librosa cache can fail in containerized environments.
        # We explicitly disable it to prevent the '__o_fold' error.
        import librosa.core
        librosa.core.cache_level = 0 # Disable cache
        # Suppress the specific warning that might still appear
        warnings.filterwarnings("ignore", message=".*cannot cache function '__o_fold'.*")
        # -----------------------------------------------------------

        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None)
        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)
        # Simple word count estimate (assuming average speaking rate)
        # A more robust method would involve ASR word timestamps
        words = transcribe_audio(audio_file_path).split()
        word_count = len(words)
        pace = word_count / duration if duration > 0 else 0 # words per second

        # Count filler words (basic example)
        filler_words = ["um", "uh", "like", "you know", "so", "basically"]
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        return {"pace": f"{pace:.2f} words/sec", "filler_words": filler_count}
    except Exception as e:
        error_msg = f"Error: {e}"
        print(f"Error analyzing audio features: {e}")
        # Returning the error message can help diagnose UI issues faster
        return {"pace": error_msg, "filler_words": error_msg}

def generate_feedback(question, answer, role):
    """Generates feedback using the LLM."""
    if not answer.strip():
        return "Please provide an answer to receive feedback."

    # Define the prompt for the LLM
    prompt = f"""
You are an experienced career coach providing constructive feedback for a job interview practice session.

Job Role: {role}
Interview Question: {question}
Candidate's Answer: {answer}

Please provide feedback structured as follows:
1.  **Content Analysis:** Evaluate the relevance, depth, and accuracy of the answer in relation to the question and job role.
2.  **Structure (STAR):** Assess if the answer follows a clear structure (Situation, Task, Action, Result) where applicable.
3.  **Clarity:** Comment on how clear and concise the answer was.
4.  **Suggestions for Improvement:** Offer 1-2 specific suggestions to make the answer stronger.

Keep the feedback professional, encouraging, and actionable. Do not mention this prompt or your role as an AI.
"""
    print("Generating feedback with LLM...")
    inputs = llm_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        # Adjust max_new_tokens as needed
        outputs = llm_model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True, pad_token_id=llm_tokenizer.eos_token_id)
    feedback = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Generated Feedback: {feedback}")
    return feedback


# --- Gradio Interface ---
def process_interview(role, question, audio_file):
    """Main processing function that ties everything together."""
    start_time = time.time()
    transcription = transcribe_audio(audio_file)
    audio_features = analyze_audio_features(audio_file)
    feedback = generate_feedback(question, transcription, role)
    processing_time = time.time() - start_time
    return transcription, audio_features.get("pace", "N/A"), audio_features.get("filler_words", "N/A"), feedback, f"Processing completed in {processing_time:.2f} seconds."

# Define roles for dropdown
roles = ["React",
    "JavaScript", 
    "Java",
    "Python",
    "SQL",
    "Angular",
    "Node.js",
    "AWS",
    "Docker",
    "Kubernetes",
    "Git",
    "Machine Learning",
    "Data Science",
    "DevOps",
    "Microservices",
    "REST API",
    "Cloud Computing",
    "Spring Boot",
    "Linux",
    "Agile",
    "TypeScript",
    "C#",
    "ASP.NET",
    "MongoDB",
    "PostgreSQL",
    "Redis",
    "TensorFlow",
    "PyTorch",
    "Jenkins",
    "Terraform",
    "React Native",
    "Flutter",
    "Vue.js",
    "Next.js",
    "Cybersecurity",
    "GraphQL",
    "Go",
    "Blockchain",
    "Apache Kafka",
    "Elasticsearch"]

with gr.Blocks(title="AI Interview Simulator") as demo:
    gr.Markdown("## 🎙️ AI Interview Simulator")
    gr.Markdown("Practice your interview skills and get instant AI feedback!")
    with gr.Row():
        role_dropdown = gr.Dropdown(choices=roles, label="Select Job Role", value="React")
        question_dropdown = gr.Dropdown(choices=[], label="Select Question")

    # Update questions based on role
    def update_questions(selected_role):
        questions = get_interview_questions(selected_role)
        return gr.update(choices=questions, value=questions[0] if questions else "")

    role_dropdown.change(fn=update_questions, inputs=role_dropdown, outputs=question_dropdown)

    audio_input = gr.Audio(label="Record Your Answer", type="filepath")
    transcribe_btn = gr.Button("Submit & Get Feedback")

    with gr.Column():
        transcription_output = gr.Textbox(label="Transcription")
        pace_output = gr.Textbox(label="Speaking Pace")
        filler_output = gr.Textbox(label="Filler Words Detected")
        feedback_output = gr.Textbox(label="AI Feedback", lines=10)
        status_output = gr.Textbox(label="Status")

    transcribe_btn.click(
        fn=process_interview,
        inputs=[role_dropdown, question_dropdown, audio_input],
        outputs=[transcription_output, pace_output, filler_output, feedback_output, status_output]
    )

    # Initialize questions on load
    demo.load(fn=update_questions, inputs=role_dropdown, outputs=question_dropdown)

# Launch the app, listening on all interfaces (important for Docker)
# The port must match EXPOSE in Dockerfile and app_port in README.md
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
