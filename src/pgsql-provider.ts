/**
 * @module pgsql-provider
 *
 * PostgreSQL implementation of {@link IDbProvider}.
 *
 * Uses the [`pg`](https://www.npmjs.com/package/pg) package (Client) for
 * connection management and query execution.
 *
 * SQL Dialect:
 * - Identifiers quoted with double quotes: `"columnName"`
 * - Positional parameters: `$1`, `$2`, ...
 * - `CREATE TABLE IF NOT EXISTS` for idempotent DDL
 */

import { Client } from 'pg';
import {
  applyClientSideQuery,
  buildDeleteSql,
  buildFindSql,
  buildInsertSql,
  buildSelectAllSql,
  buildSelectSql,
  buildUpdateSql,
  IDbProvider,
  QueryObject,
  SqlDialect,
  TableColumnInfo
} from '@romatech/orm';

/**
 * Configuration object for a PostgreSQL connection.
 *
 * Accepts either a libpq connection string / DSN or a structured object.
 * The object form is recommended for production because it keeps credentials
 * out of string interpolation and avoids shell-escaping issues.
 *
 * @example
 * // Connection string (libpq DSN) form
 * const config: PgSqlConfig = 'postgresql://user:secret@localhost:5432/mydb';
 *
 * @example
 * // Object form
 * const config: PgSqlConfig = {
 *   host: 'localhost',
 *   port: 5432,
 *   user: 'app_user',
 *   password: 'secret',
 *   database: 'mydb'
 * };
 */
type PgSqlConfig = string | {
  /** Hostname or IP address of the PostgreSQL server. */
  host: string;
  /** TCP port (default: 5432). */
  port?: number;
  /** PostgreSQL role name for authentication. */
  user: string;
  /** PostgreSQL role password. */
  password: string;
  /** Target database name. */
  database: string;
};

/**
 * SQL dialect definition for PostgreSQL.
 *
 * - **Identifier quoting**: wraps identifiers in double-quotes (`"name"`) and
 *   escapes embedded double-quotes by doubling them (`""`), following the SQL
 *   standard and PostgreSQL conventions.
 * - **Parameter style**: uses `$1`, `$2`, … (1-based index) which is the
 *   native placeholder syntax for the `pg` driver.
 */
const dialect: SqlDialect = {
  quoteIdentifier: identifier => `"${identifier.replace(/"/g, '""')}"`,
  parameter: index => `$${index + 1}`
};

/**
 * RomaTech ORM database provider for **PostgreSQL** (v12+).
 *
 * Wraps the official `pg` (node-postgres) `Client` for async/await operation.
 * A single client connection is maintained per provider instance.  All DML
 * and DDL statements use parameterized queries with `$N` placeholders to
 * prevent SQL injection.
 *
 * Implements {@link IDbProvider} — the common interface shared by all
 * RomaTech ORM providers.
 *
 * @example
 * import { PgSqlProvider } from '@romatech/orm-providers-pgsql';
 * import { DbContext, entity, primaryKey } from '@romatech/orm';
 *
 * \@entity('orders')
 * class Order {
 *   \@primaryKey()
 *   id!: number;
 *   customerId!: number;
 *   total!: number;
 * }
 *
 * const provider = new PgSqlProvider({
 *   host: 'localhost',
 *   user: 'postgres',
 *   password: 'secret',
 *   database: 'shop'
 * });
 *
 * const ctx = new DbContext(provider);
 * await ctx.connect();
 * const orders = ctx.set(Order);
 * await orders.addAsync({ id: 1, customerId: 7, total: 99.50 });
 * await ctx.disconnect();
 */
export class PgSqlProvider implements IDbProvider {
  /** The `pg` Client used for all database operations. */
  private client!: Client;

  /**
   * Creates a new `PgSqlProvider` instance.
   *
   * No TCP connection is established until {@link connect} is called.
   *
   * @param config - Either a libpq connection string or a structured
   *   {@link PgSqlConfig} object.
   */
  constructor(private config: PgSqlConfig) {}

  /**
   * Opens a connection to the PostgreSQL server.
   *
   * Instantiates a `pg.Client` and calls its `connect()` method.  When
   * `connectionString` is supplied it overrides the config passed to the
   * constructor, allowing the ORM framework to inject a runtime connection
   * string (e.g., from environment variables).
   *
   * @param connectionString - Optional override connection URI / DSN.
   * @returns A promise that resolves when the connection is ready.
   * @throws {Error} If the server is unreachable or credentials are invalid.
   *
   * @example
   * await provider.connect();
   * await provider.connect('postgresql://user:pass@prod-host/mydb');
   */
  async connect(connectionString = ''): Promise<void> {
    this.client = new Client(connectionString || this.config);
    await this.client.connect();
  }

  /**
   * Closes the connection to the PostgreSQL server.
   *
   * Sends a termination message to the server and closes the underlying socket.
   *
   * @returns A promise that resolves once the connection is closed.
   *
   * @example
   * await provider.disconnect();
   */
  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.end();
    }
  }

  /**
   * Inserts a single entity into the specified table.
   *
   * Delegates to {@link buildInsertSql} which produces
   * `INSERT INTO "tableName" ("col1", "col2", …) VALUES ($1, $2, …)`.
   * Properties with `undefined` values are excluded so column defaults (e.g.
   * `DEFAULT gen_random_uuid()`) are respected.
   *
   * @param entity - Plain object whose own enumerable properties map to table
   *   columns.
   * @param tableName - Target table name (will be double-quote-quoted).
   * @returns A promise that resolves when the row has been inserted.
   * @throws {Error} On constraint violations (unique, NOT NULL, FK, etc.).
   *
   * @example
   * await provider.add({ id: 1, customerId: 7, total: 99.50 }, 'orders');
   */
  async add<T extends object>(entity: T, tableName: string): Promise<void> {
    const command = buildInsertSql(tableName, entity, dialect);
    await this.executeNonQuery(command.sql, command.params);
  }

  /**
   * Inserts multiple entities into the specified table sequentially.
   *
   * @param entities - Array of entities to insert.
   * @param tableName - Target table name.
   * @returns A promise that resolves when all rows have been inserted.
   * @throws {Error} If any individual insert fails; previous inserts in the
   *   same call are not rolled back automatically.
   *
   * @example
   * await provider.addRange(
   *   [{ id: 1, customerId: 7, total: 99.50 }, { id: 2, customerId: 8, total: 45 }],
   *   'orders'
   * );
   */
  async addRange<T extends object>(entities: T[], tableName: string): Promise<void> {
    for (const entity of entities) {
      await this.add(entity, tableName);
    }
  }

  /**
   * Updates an existing row identified by its primary key.
   *
   * Introspects column metadata to find the primary key, then generates
   * `UPDATE "tableName" SET "col1" = $1, … WHERE "pk" = $N`.
   * If the entity contains only the primary-key field, the operation is
   * skipped.
   *
   * @param entity - Object whose primary-key property identifies the row and
   *   remaining properties supply the new values.
   * @param tableName - Target table name.
   * @returns A promise that resolves when the row has been updated.
   * @throws {Error} If no matching row exists or a constraint is violated.
   *
   * @example
   * await provider.update({ id: 1, total: 110.00 }, 'orders');
   */
  async update<T extends object>(entity: T, tableName: string): Promise<void> {
    const primaryKey = await this.getPrimaryKeyColumn(tableName);
    // Guard: skip round-trip if there are no columns to update besides the PK.
    if (!Object.keys(entity).some(key => key !== primaryKey)) {
      return;
    }
    const command = buildUpdateSql(tableName, entity, primaryKey, dialect);
    await this.executeNonQuery(command.sql, command.params);
  }

  /**
   * Deletes a single row identified by the entity's primary key.
   *
   * Generates `DELETE FROM "tableName" WHERE "pk" = $1`.
   *
   * @param entity - Object whose primary-key property identifies the row to
   *   delete.
   * @param tableName - Target table name.
   * @returns A promise that resolves when the row has been deleted.
   * @throws {Error} On foreign-key constraint violations.
   *
   * @example
   * await provider.remove({ id: 1 }, 'orders');
   */
  async remove<T extends object>(entity: T, tableName: string): Promise<void> {
    const command = buildDeleteSql(tableName, entity, await this.getPrimaryKeyColumn(tableName), dialect);
    await this.executeNonQuery(command.sql, command.params);
  }

  /**
   * Deletes multiple rows, each identified by its entity's primary key.
   *
   * @param entities - Array of entities whose primary keys identify the rows
   *   to delete.
   * @param tableName - Target table name.
   * @returns A promise that resolves when all rows have been deleted.
   * @throws {Error} If any individual delete fails.
   *
   * @example
   * await provider.removeRange([{ id: 1 }, { id: 2 }], 'orders');
   */
  async removeRange<T extends object>(entities: T[], tableName: string): Promise<void> {
    for (const entity of entities) {
      await this.remove(entity, tableName);
    }
  }

  /**
   * Retrieves a single row by its primary key.
   *
   * Generates `SELECT * FROM "tableName" WHERE "pk" = $1` and returns the
   * first result, or `undefined` when no matching row is found.
   *
   * @param entity - Object whose primary-key property supplies the lookup
   *   value.
   * @param tableName - Target table name.
   * @returns The matching row cast to `T`, or `undefined` if not found.
   * @throws {Error} On connection or query errors.
   *
   * @example
   * const order = await provider.find({ id: 42 }, 'orders');
   * if (order) console.log(order.total);
   */
  async find<T extends object>(entity: T, tableName: string): Promise<T | undefined> {
    const command = buildFindSql(tableName, entity, await this.getPrimaryKeyColumn(tableName), dialect);
    const rows = await this.executeQuery<T>(command.sql, command.params);
    return rows[0];
  }

  /**
   * Returns all rows from the specified table.
   *
   * Generates `SELECT * FROM "tableName"`.  For large tables, prefer
   * {@link executeQuery} with a filtered {@link QueryObject}.
   *
   * @param tableName - Source table name.
   * @returns An array of all rows cast to `T`.
   * @throws {Error} On connection or query errors.
   *
   * @example
   * const all = await provider.getAll<Order>('orders');
   */
  async getAll<T>(tableName: string): Promise<T[]> {
    return this.executeQuery<T>(buildSelectAllSql(tableName, dialect));
  }

  /**
   * No-op for this provider.
   *
   * PostgreSQL statements are auto-committed by default on a plain client
   * (no explicit `BEGIN` / `COMMIT`).
   *
   * @returns A resolved promise.
   */
  async saveChanges(): Promise<void> {
    return;
  }

  /**
   * Records a migration entry in the `"__roma_migrations"` history table.
   *
   * Creates the table if it does not yet exist, then inserts a row with the
   * migration name and its SQL script.
   *
   * @param migrationName - Unique name for the migration (e.g.
   *   `"20240101_CreateOrders"`).
   * @param migrationScript - The full DDL/DML script applied by this
   *   migration.
   * @returns A promise that resolves once the record is persisted.
   * @throws {Error} On duplicate migration name (PRIMARY KEY violation).
   *
   * @example
   * await provider.addMigration('20240101_Init', 'CREATE TABLE orders (...)');
   */
  async addMigration(migrationName: string, migrationScript: string): Promise<void> {
    await this.ensureMigrationHistoryTable();
    await this.executeNonQuery(
      'INSERT INTO "__roma_migrations" ("migrationName", "migrationScript") VALUES ($1, $2)',
      [migrationName, migrationScript]
    );
  }

  /**
   * Removes a migration entry from the `"__roma_migrations"` history table.
   *
   * Used during a downgrade to erase the record of an applied migration.
   *
   * @param migrationName - The name of the migration to remove.
   * @returns A promise that resolves once the record has been deleted.
   *
   * @example
   * await provider.removeMigration('20240101_Init');
   */
  async removeMigration(migrationName: string): Promise<void> {
    await this.ensureMigrationHistoryTable();
    await this.executeNonQuery('DELETE FROM "__roma_migrations" WHERE "migrationName" = $1', [migrationName]);
  }

  /**
   * No-op for this provider — migrations are applied individually via the CLI.
   * @returns A resolved promise.
   */
  async applyMigrations(): Promise<void> {
    return;
  }

  /**
   * Returns the list of migration names recorded in the history table.
   *
   * Delegates to {@link getMigrationHistory}.
   *
   * @returns An array of migration name strings in ascending alphabetical order.
   *
   * @example
   * const applied = await provider.getMigrations();
   */
  async getMigrations(): Promise<string[]> {
    return this.getMigrationHistory();
  }

  /**
   * Queries `"__roma_migrations"` for all previously applied migration names.
   *
   * Creates the history table first if it does not exist, making this safe to
   * call on a fresh database.
   *
   * @returns An array of migration name strings ordered alphabetically.
   * @throws {Error} On connection or query errors.
   *
   * @example
   * const history = await provider.getMigrationHistory();
   */
  async getMigrationHistory(): Promise<string[]> {
    await this.ensureMigrationHistoryTable();
    const rows = await this.executeQuery<{ migrationName: string }>(
      'SELECT "migrationName" FROM "__roma_migrations" ORDER BY "migrationName"'
    );
    return rows.map(row => row.migrationName);
  }

  /**
   * No-op for this provider — handled by the CLI.
   * @returns A resolved promise.
   */
  async updateDatabase(_targetMigration?: string): Promise<void> {
    return;
  }

  /**
   * No-op for this provider — handled by the CLI.
   * @returns A resolved promise.
   */
  async downgradeDatabase(_targetMigration?: string): Promise<void> {
    return;
  }

  /**
   * Creates a table in PostgreSQL if it does not already exist.
   *
   * Uses `CREATE TABLE IF NOT EXISTS`, which is natively supported by
   * PostgreSQL (v9.1+) and makes the operation idempotent.
   *
   * @param input.tableName - Name of the table to create.
   * @param input.columns - Column definitions; each column's `tsType` is
   *   mapped to a PostgreSQL type via {@link mapColumnType}.
   * @param input.primaryKey - Optional explicit primary-key column name.
   *   Falls back to the first column with `primaryKey: true`.
   * @returns A promise that resolves once the table exists.
   * @throws {Error} On SQL syntax or permission errors.
   *
   * @example
   * await provider.createTable({
   *   tableName: 'orders',
   *   columns: [
   *     { name: 'id', tsType: 'number', primaryKey: true },
   *     { name: 'total', tsType: 'number' }
   *   ]
   * });
   */
  async createTable(input: { tableName: string; columns: TableColumnInfo[]; primaryKey?: string }): Promise<void> {
    const primaryKey = input.primaryKey || input.columns.find(column => column.primaryKey)?.name;
    const columns = input.columns
      .map(column => `${dialect.quoteIdentifier(column.name)} ${this.mapColumnType(column)}${column.primaryKey ? ' NOT NULL' : ''}`)
      .join(', ');
    const primaryKeySql = primaryKey ? `, PRIMARY KEY (${dialect.quoteIdentifier(primaryKey)})` : '';
    await this.executeNonQuery(`CREATE TABLE IF NOT EXISTS ${dialect.quoteIdentifier(input.tableName)} (${columns}${primaryKeySql})`);
  }

  /**
   * Drops a table from the database if it exists.
   *
   * Uses `DROP TABLE IF EXISTS`, which is idempotent and does not error when
   * the table is absent.
   *
   * @param tableName - Name of the table to drop.
   * @returns A promise that resolves once the table has been dropped.
   * @throws {Error} On foreign-key constraint violations unless `CASCADE` is
   *   used.
   *
   * @example
   * await provider.dropTable('orders');
   */
  async dropTable(tableName: string): Promise<void> {
    await this.executeNonQuery(`DROP TABLE IF EXISTS ${dialect.quoteIdentifier(tableName)}`);
  }

  /**
   * Adds a new column to an existing table.
   *
   * Generates `ALTER TABLE "tableName" ADD COLUMN "columnName" <SQL type>`.
   *
   * @param tableName - Name of the table to alter.
   * @param column - Column definition including name and TypeScript type.
   * @returns A promise that resolves once the column has been added.
   * @throws {Error} If the column already exists.
   *
   * @example
   * await provider.addColumn('orders', { name: 'note', tsType: 'string' });
   */
  async addColumn(tableName: string, column: TableColumnInfo): Promise<void> {
    await this.executeNonQuery(
      `ALTER TABLE ${dialect.quoteIdentifier(tableName)} ADD COLUMN ${dialect.quoteIdentifier(column.name)} ${this.mapColumnType(column)}`
    );
  }

  /**
   * Removes a column from an existing table.
   *
   * Generates `ALTER TABLE "tableName" DROP COLUMN "columnName"`.
   *
   * @param tableName - Name of the table to alter.
   * @param columnName - Name of the column to drop.
   * @returns A promise that resolves once the column has been removed.
   * @throws {Error} If the column is referenced by a view, constraint, or
   *   index.
   *
   * @example
   * await provider.removeColumn('orders', 'note');
   */
  async removeColumn(tableName: string, columnName: string): Promise<void> {
    await this.executeNonQuery(
      `ALTER TABLE ${dialect.quoteIdentifier(tableName)} DROP COLUMN ${dialect.quoteIdentifier(columnName)}`
    );
  }

  /**
   * No-op for this provider — scaffold is handled by the CLI command.
   * @returns A resolved promise.
   */
  async scaffold(_connectionString: string): Promise<void> {
    return;
  }

  /**
   * Returns the names of all tables in the `public` schema.
   *
   * Queries `pg_tables` filtering to `schemaname = 'public'`.  Tables in
   * other schemas are not included.
   *
   * @returns An array of table name strings.
   * @throws {Error} On connection or query errors.
   *
   * @example
   * const tables = await provider.getTables();
   * console.log(tables); // ['orders', 'users', '__roma_migrations']
   */
  async getTables(): Promise<string[]> {
    const rows = await this.executeQuery<{ tablename: string }>(`
      SELECT tablename FROM pg_tables
      WHERE schemaname = 'public'
    `);
    return rows.map(row => row.tablename);
  }

  /**
   * Returns column metadata for a given table, including primary-key
   * detection.
   *
   * Joins `information_schema.columns` with `information_schema.table_constraints`
   * and `information_schema.key_column_usage` to compute an `EXISTS` subquery
   * that flags primary-key columns.
   *
   * @param table - Name of the table to inspect.
   * @returns An array of {@link TableColumnInfo} objects with name, tsType,
   *   and primaryKey flag.
   * @throws {Error} On connection or query errors.
   *
   * @example
   * const cols = await provider.getColumnsForTable('orders');
   * // [{ name: 'id', tsType: 'number', primaryKey: true }, ...]
   */
  async getColumnsForTable(table: string): Promise<TableColumnInfo[]> {
    const rows = await this.executeQuery<{ name: string; type: string; primary_key: boolean }>(
      `
      SELECT c.column_name as name, c.data_type as type,
        EXISTS (
          SELECT 1
          FROM information_schema.table_constraints tc
          JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
          WHERE tc.constraint_type = 'PRIMARY KEY'
            AND tc.table_name = $1
            AND kcu.column_name = c.column_name
        ) AS primary_key
      FROM information_schema.columns c
      WHERE c.table_name = $1
      `,
      [table]
    );

    return rows.map(column => ({
      name: column.name,
      primaryKey: column.primary_key,
      tsType: this.mapDbTypeToTsType(column.type)
    }));
  }

  /**
   * Executes a SQL query or a structured {@link QueryObject} and returns the
   * result rows.
   *
   * **Overload 1 — raw SQL:**
   * ```ts
   * const rows = await provider.executeQuery<Order>(
   *   'SELECT * FROM "orders" WHERE "id" = $1',
   *   [42]
   * );
   * ```
   *
   * **Overload 2 — QueryObject:**
   * ```ts
   * const rows = await provider.executeQuery('orders', query);
   * ```
   * When a `QueryObject` is passed the provider pushes serializable `WHERE`
   * and `ORDER BY` to PostgreSQL, then applies remaining client-side predicates
   * via {@link applyClientSideQuery}.
   *
   * @param query - Either a raw SQL string or a table name (for QueryObject).
   * @param params - Parameter array for raw SQL, or a `QueryObject`.
   * @returns A promise resolving to an array of result rows.
   * @throws {Error} On SQL syntax errors or connection failures.
   */
  async executeQuery<T = any>(query: string, params?: any[]): Promise<T[]>;
  async executeQuery<T, TResult = T>(entityName: string, query: QueryObject<T, TResult>): Promise<TResult[]>;
  async executeQuery<T, TResult = T>(
    queryOrEntityName: string,
    paramsOrQuery: any[] | QueryObject<T, TResult> = []
  ): Promise<T[] | TResult[]> {
    if (!Array.isArray(paramsOrQuery)) {
      // QueryObject path: build server-side SQL then apply remaining
      // client-side predicates (e.g. JS closures that can't be serialized).
      const command = buildSelectSql(queryOrEntityName, paramsOrQuery, dialect);
      const rows = await this.executeQuery<T>(command.sql, command.params);
      return applyClientSideQuery(rows, paramsOrQuery);
    }

    // Raw SQL path: pg's Client.query() returns { rows: T[] }.
    const result = await this.client.query(queryOrEntityName, paramsOrQuery);
    return result.rows as T[];
  }

  /**
   * Executes a non-query SQL statement (INSERT / UPDATE / DELETE / DDL).
   *
   * Uses the `pg` client's `query()` method with `$N` parameterized inputs.
   *
   * @param sql - Parameterized SQL statement with `$1`, `$2`, … placeholders.
   * @param params - Positional parameter values.
   * @returns A promise that resolves when the statement completes.
   * @throws {Error} On SQL errors or connection failures.
   *
   * @example
   * await provider.executeNonQuery(
   *   'UPDATE "orders" SET "total" = $1 WHERE "id" = $2',
   *   [110.00, 1]
   * );
   */
  async executeNonQuery(sql: string, params: any[] = []): Promise<void> {
    await this.client.query(sql, params);
  }

  /**
   * Creates the `"__roma_migrations"` history table if it does not exist.
   *
   * Uses `CREATE TABLE IF NOT EXISTS` for idempotency.  The schema stores
   * the migration name (PK), the full SQL script (`TEXT`), and an `appliedAt`
   * timestamptz defaulting to `now()`.
   */
  private async ensureMigrationHistoryTable(): Promise<void> {
    await this.executeNonQuery(`
      CREATE TABLE IF NOT EXISTS "__roma_migrations" (
        "migrationName" TEXT NOT NULL PRIMARY KEY,
        "migrationScript" TEXT NOT NULL,
        "appliedAt" TIMESTAMPTZ NOT NULL DEFAULT now()
      )
    `);
  }

  /**
   * Resolves the primary-key column name for a given table.
   *
   * Falls back to `'id'` when no primary key is detected (e.g. composite keys
   * or views).
   *
   * @param tableName - Table to inspect.
   * @returns The primary-key column name or `'id'`.
   */
  private async getPrimaryKeyColumn(tableName: string): Promise<string> {
    const primaryKey = (await this.getColumnsForTable(tableName)).find(column => column.primaryKey)?.name;
    return primaryKey || 'id';
  }

  /**
   * Maps a {@link TableColumnInfo} TypeScript type to a PostgreSQL column type.
   *
   * | tsType      | PostgreSQL type     |
   * |-------------|---------------------|
   * | `number`    | `DOUBLE PRECISION`  |
   * | `boolean`   | `BOOLEAN`           |
   * | `Date`      | `TIMESTAMPTZ`       |
   * | `unknown`   | `JSONB`             |
   * | *(default)* | `TEXT`              |
   *
   * @param column - Column definition including `tsType`.
   * @returns The PostgreSQL column type string.
   */
  private mapColumnType(column: TableColumnInfo): string {
    const type = column.tsType.toLowerCase();
    // Use DOUBLE PRECISION for all numeric types (both PK and non-PK).
    if (type.includes('number')) return 'DOUBLE PRECISION';
    if (type.includes('boolean')) return 'BOOLEAN';
    // TIMESTAMPTZ stores both date and time with UTC time-zone offset.
    if (type.includes('date')) return 'TIMESTAMPTZ';
    // JSONB enables efficient querying of JSON documents.
    if (type.includes('unknown')) return 'JSONB';
    // TEXT is the idiomatic unbounded string type in PostgreSQL.
    return 'TEXT';
  }

  /**
   * Maps a PostgreSQL `data_type` string to the corresponding TypeScript type
   * used in scaffolded entity classes.
   *
   * | PostgreSQL type pattern                    | TypeScript type |
   * |--------------------------------------------|-----------------|
   * | `int`, `decimal`, `numeric`, `float`, `double`, `real`, `serial` | `number` |
   * | `bool`                                     | `boolean`       |
   * | `date`, `time` (any variant)               | `Date`          |
   * | `json`, `array`                            | `unknown`       |
   * | *(anything else)*                          | `string`        |
   *
   * @param type - Raw `data_type` string from `information_schema.columns`.
   * @returns A TypeScript type name string.
   */
  private mapDbTypeToTsType(type: string): string {
    const normalized = type.toLowerCase();
    if (/(int|decimal|numeric|float|double|real|serial)/.test(normalized)) return 'number';
    if (/(bool)/.test(normalized)) return 'boolean';
    if (/(date|time)/.test(normalized)) return 'Date';
    if (/(json|array)/.test(normalized)) return 'unknown';
    return 'string';
  }
}
