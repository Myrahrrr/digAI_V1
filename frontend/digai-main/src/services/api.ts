export default async function api(url: string, options: any) {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
    },
  });
  return await response.json();
}
