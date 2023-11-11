import { API_ADDR, fetchWithTimeout } from '@/lib/api/utils';
import { translationResult } from '@/lib/model/text2sql/type';

export async function getTranslationResult(dbName: string, question: string): Promise<translationResult> {
    const addr = `${API_ADDR}/model/text2sql/?dbName=${dbName}&question=${question}`;
    return fetchWithTimeout(addr).then(res => res.json());
}
