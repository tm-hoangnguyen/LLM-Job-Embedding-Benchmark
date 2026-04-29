import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# BASE_DIR = os.path.dirname((os.path.abspath(__file__)))
# dotenv_path = os.path.join(BASE_DIR, '.env.dev.llm')

# load dotenv
# load_dotenv(dotenv_path)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.dev.llm"))

import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="scaleway",
    api_key=os.getenv("QWEN_API_KEY"),
)

result = client.feature_extraction(
    '''About the job
Google's software engineers develop the next-generation technologies that change how billions of users connect, explore, and interact with information and one another. Our products need to handle information at massive scale, and extend well beyond web search. We're looking for engineers who bring fresh ideas from all areas, including information retrieval, distributed computing, large-scale system design, networking and data storage, security, artificial intelligence, natural language processing, UI design and mobile; the list goes on and is growing every day. As a software engineer, you will work on a specific project critical to Google’s needs with opportunities to switch teams and projects as you and our fast-paced business grow and evolve. We need our engineers to be versatile, display leadership qualities and be enthusiastic to take on new problems across the full-stack as we continue to push technology forward.

With your technical expertise you will manage project priorities, deadlines, and deliverables. You will design, develop, test, deploy, maintain, and enhance software solutions.
The AI and Infrastructure team works on the world’s toughest problems, redefining what’s possible and the possible easy. We empower Google customers by delivering AI and Infrastructure at unparalleled scale, efficiency, reliability and velocity. Our customers include Googlers, Googler Cloud customers, and billions of Google users worldwide. We’re at the center of amazing work at Google by being the “flywheel” that enables our advanced AI models, delivers computing power across global services, and offers platforms that developers use to build services.

In AI and Infrastructure, we shape the future of hyperscale computing by inventing and creating world-leading future technology, and drive global impact by contributing to Google infrastructure, from software to hardware (including building Vertex AI for Google Cloud). We work on complex technologies at a global scale with key players in the AI and systems space. Join a team of talented individuals who not only work together to keep data centers operating efficiently but also create a legacy of driving innovation by building some of the most complex systems technologies.

Responsibilities
Write product or system development code.
Participate in, or lead design reviews with peers and stakeholders to decide amongst available technologies.
Review code developed by other developers and provide feedback to ensure best practices (e.g., style guidelines, checking code in, accuracy, testability, and efficiency).
Contribute to existing documentation or educational content and adapt content based on product/program updates and user feedback.
Triage product or system issues and debug/track/resolve by analyzing the sources of issues and the impact on hardware, network, or service operations and quality.''',
    model="Qwen/Qwen3-Embedding-8B",
)

print(len(result[0]))