
export async function POST(req: Request) {
  const { username, password } = await req.json();

  console.log("Received:", username, password);
  console.log("Expected:", process.env.USERNAME, process.env.PASSWORD);

  if (
    username === username &&
    password === password
  ) {
    return Response.json({ success: true });
  }

  return Response.json({ success: false }, { status: 401 });
}
