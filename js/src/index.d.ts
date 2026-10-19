export type FieldType =
  | 'text'
  | 'number'
  | 'boolean'
  | 'enum'
  | 'date'
  | 'time'
  | 'datetime'
  | 'duration'
  | 'list'
  | 'estimate'
  | 'forecast';

export interface FieldSpec {
  type: FieldType;
  instructions?: string;
  unit?: string;
  decimals?: number;
  integer?: boolean;
  items?: string;
  options?: string[];
  constraint?: string;
  min?: number;
  max?: number;
  step?: number;
  scale?: 'linear' | 'log';
  percentiles?: boolean;
  horizon?: number;
  interval?: 'hour' | 'day' | 'week' | 'month';
}

export interface FieldAnswer {
  type: FieldType;
  value: any;
  p_null?: number;
  p_true?: number;
  span?: [number, number];
  spans?: Array<[number, number]>;
  options?: string[];
  probs?: Record<string, number>;
  guess?: any;
  confidence?: number;
  span_entropy_bits?: number;
  abstention_reason?: string;
  verified?: boolean;
  unit?: string;
  median?: number | number[];
  p10?: number | number[];
  p90?: number | number[];
  p_below?: number;
  p_above?: number;
  percentiles?: Record<string, number>;
}

export interface ExtractionResult {
  model: string;
  invariants_passed: boolean;
  latency_ms: number;
  answers: Record<string, FieldAnswer>;
}

export declare class NullspanExtractor {
  constructor(fields: Record<string, FieldSpec>, options?: Record<string, any>);
  run(state: string): ExtractionResult;
  batch(states: string[]): ExtractionResult[];
}

export declare function extract(
  state: string,
  fields: Record<string, FieldSpec>,
  options?: Record<string, any>
): ExtractionResult;

export declare function extract(
  fields: Record<string, FieldSpec>,
  options?: Record<string, any>
): NullspanExtractor;

export declare function extract(payload: {
  state: string;
  fields: Record<string, FieldSpec>;
}): ExtractionResult;
