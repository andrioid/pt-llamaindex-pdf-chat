import {
  ContextChatEngine,
  FunctionTool,
  HuggingFaceEmbedding,
  Ollama,
  Settings,
  SimpleDirectoryReader,
  storageContextFromDefaults,
  VectorStoreIndex,
} from "llamaindex";

import readline from "node:readline/promises";

// Your imports go here
/*
Settings.llm = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  additionalSessionOptions: {
    //baseURL: "https://api.model.box/v1",
  },
  model: "gpt-4o",
});
*/
Settings.llm = new Ollama({
  model: "llama3.2:1b",
});
Settings.callbackManager.on("llm-tool-call", (event) => {
  console.log(event.detail);
});
Settings.callbackManager.on("llm-tool-result", (event) => {
  console.log(event.detail);
});
Settings.embedModel = new HuggingFaceEmbedding({
  modelType: "BAAI/bge-small-en-v1.5",
  modelOptions: {
    dtype: "fp32", // ???
  },
});

const sumNumbers = ({ a, b }: { a: number; b: number }) => {
  return `${a + b}`;
};

const tool = FunctionTool.from(sumNumbers, {
  name: "sumNumbers",
  description: "Use this function to sum two numbers",
  parameters: {
    type: "object",
    properties: {
      a: {
        type: "number",
        description: "First number to sum",
      },
      b: {
        type: "number",
        description: "Second number to sum",
      },
    },
    required: ["a", "b"],
  },
});

const storageContext = await storageContextFromDefaults({
  persistDir: "./storage",
});

async function main() {
  // the rest of your code goes here
  const reader = new SimpleDirectoryReader();
  const documents = await reader.loadData("./data");

  const index = await VectorStoreIndex.fromDocuments(documents, {
    storageContext,
  });
  const retriever = await index.asRetriever({
    similarityTopK: 0.75,
    //topK: 5
  });

  const chatEngine = new ContextChatEngine({
    retriever,
    systemPrompt: `You are a helpful assistant. You must answer strictly based on provided context. Provide answers in a short an concise way. If answer can't be found in the provided documents, prespond with: "I don't know. It's not in my files"`,
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const query = await rl.question("Query: ");
    const response = await chatEngine.chat({
      message: query,
      stream: false,
    });
    if (!response.sourceNodes || response.sourceNodes.length === 0) {
      process.stdout.write("Irrelevant question, ask me about D&D\n");
      continue;
    }
    const answer = response.message.content.toString();
    if (!answer) {
      process.stdout.write("I don't know, sorry\n");
      continue;
    }
    console.log(
      "ANSWER...",
      response.sourceNodes.map((sn) => sn.score)
    );
    //console.debug("source nodes", response.sourceNodes);

    process.stdout.write(answer);
  }
}

main().catch(console.error);
