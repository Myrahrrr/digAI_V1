import api from "./api";

const chatbotURL = import.meta.env.VITE_CHATBOT_API;

const ChatbotService = {
  init: async (id: string, prolific: string) => {
    return await api(chatbotURL + "/init", {
      method: "POST",
      body: JSON.stringify({ id, prolific }),
    });
  },
  navigate: async (id: string, page: string) => {
    return await api(chatbotURL + "/navigate", {
      method: "POST",
      body: JSON.stringify({ id: id, page: page }),
    });
  },

  chat: async (id: string, utterance: string, lang: string) => {
    return await api(chatbotURL + "/chat", {
      method: "POST",
      body: JSON.stringify({ id, utterance, lang }),
    });
  },
  like: async (id: string, product: string) => {
    return await api(chatbotURL + "/like", {
      method: "POST",
      body: JSON.stringify({ id: id, product: product }),
    });
  },
  dislike: async (id: string, product: string) => {
    return await api(chatbotURL + "/dislike", {
      method: "POST",
      body: JSON.stringify({ id: id, product: product }),
    });
  },
  finish: async (id: string, product: string) => {
    return await api(chatbotURL + "/finish", {
      method: "POST",
      body: JSON.stringify({ id: id, product: product }),
    });
  },
  questionnaire: async (id: string, data: any, complete: boolean) => {
    return await api(chatbotURL + "/questionnaire", {
      method: "POST",
      body: JSON.stringify({ id: id, data: data, complete: complete }),
    });
  },
};

export default ChatbotService;
