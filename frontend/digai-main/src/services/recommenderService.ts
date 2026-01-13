import api from "./api";

const recommenderURL = import.meta.env.VITE_RECOMM_API;

const RecommenderService = {
  recomm: async (state: any) => {
    return await api(recommenderURL + "/recomm", {
      method: "POST",
      body: JSON.stringify({ state }),
    });
  },
  entropy: () => {},
};

export default RecommenderService;
