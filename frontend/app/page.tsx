
// app/page.tsx

'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    const auth = localStorage.getItem('auth');
    if (auth === 'true') {
      router.replace('/protected/proposal_generator');
    } else {
      router.replace('/login');
    }
  }, [router]);

  return null;
}
