'use client';
import { getModelResult } from '@/lib/api/frontend/model';
import { modelResult } from '@/lib/model/type';
import useSWR from 'swr';

export function useTranslatedSQLByQuestion(dbName: string, question: string) {
    // Define the key for SWR based on the dbName and query.
    // If either is not present, use null to avoid fetching.
    const swrKey = dbName && question ? ['getQueryResult', dbName, question] : null;

    // Define the fetcher function directly using getQueryResult, avoiding the use of useSWRWrapper.
    const fetcher = swrKey ? () => getModelResult(dbName, question) : null;

    // Call useSWR at the top level, passing the key and fetcher.
    const { data, error } = useSWR<modelResult>(swrKey, fetcher);

    // isLoading is not directly provided by useSWR, so it must be derived from `data` and `error`.
    const isLoading = !data && !error;

    return {
        data,
        isLoading,
        isError: !!error, // Convert error to a boolean indicating if there is an error
    };
}